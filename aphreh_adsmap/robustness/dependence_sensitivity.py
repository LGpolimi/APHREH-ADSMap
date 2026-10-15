"""RS Analysis 5: temporal-dependence and block-length sensitivity.

Dependence-only realizations reuse the reference chronological differential and
hazard-weight panels. Only dependence calibration, synchronized moving-block
bootstrap, pooling and territory-level metrics are recomputed. Full-sample
yearly RCs are checked against the reference realization and must remain
numerically invariant.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import csv, hashlib, math, shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import validate
from ..engine import FixedParametersEngine, cell_key, cell_scope, finite_mean
from ..randomness import rng_for_identity
from ..storage import Store, atomic_json, clean, dumps, mapped
from .. import numerics as nm
from ..support import active_support_policy, apply_baseline_support_filter, evaluate_postbootstrap_support
from .models import CheckpointState, ComparisonMode, FactorKind, InvalidationStage, RealizationSpec, RealizationState
from .planner import FactorSpec, RealizationPlanner, ScenarioSpec, materialize_realization
from .persistence import RobustnessPersistence

ANALYSIS_ID = "ANALYSIS_5_DEPENDENCE_SENSITIVITY"
CHECKPOINT_STAGE = "RS_ANALYSIS_5"
SCHEMA_VERSION = "APHREH_RS_ANALYSIS_5_V2"

DEPENDENCE_FACTOR_REGISTRY: dict[str, dict[str, Any]] = {
    "local_acf_interval_days": {"path":"core.dependence.local_acf_interval_days","integer":True,"invalidates_from":InvalidationStage.BOOTSTRAP},
    "negligible_lag_persistence": {"path":"core.dependence.negligible_lag_persistence","integer":True,"invalidates_from":InvalidationStage.BOOTSTRAP},
    "annual_dependence_quantile": {"path":"core.dependence.annual_dependence_quantile","integer":False,"invalidates_from":InvalidationStage.BOOTSTRAP},
    "exposed_episode_quantile": {"path":"core.dependence.exposed_episode_quantile","integer":False,"invalidates_from":InvalidationStage.BOOTSTRAP},
    "final_block_length_multiplier": {"path":"core.dependence.final_block_length_multiplier","integer":False,"invalidates_from":InvalidationStage.BOOTSTRAP},
}

def _finite(v):
    try: x=float(v)
    except (TypeError,ValueError): return None
    return x if math.isfinite(x) else None

def dependence_scenarios_from_config(config: Mapping[str, Any]) -> tuple[ScenarioSpec,...]:
    section=config.get("robustness",{}).get("analysis_5_dependence_sensitivity",{})
    if not section.get("enabled",False): return ()
    ref=config["core"]["dependence"]
    out=[]
    for raw in section.get("scenarios",[]):
        factors=[]
        for name,bounds in raw.get("factors",{}).items():
            meta=DEPENDENCE_FACTOR_REGISTRY[name]
            factors.append(FactorSpec(name=name,config_path=meta["path"],analysis_id=ANALYSIS_ID,
                kind=FactorKind.NUMERICAL,invalidates_from=meta["invalidates_from"],reference_value=ref[name],
                lower=bounds["lower"],upper=bounds["upper"],integer=bool(meta["integer"])))
        out.append(ScenarioSpec(raw["scenario_id"],ANALYSIS_ID,tuple(factors),numerical_steps=int(raw["steps"]),
                                comparison_mode=ComparisonMode(raw.get("comparison_mode", "FIXED_PARAMETER")),label=raw.get("label",raw["scenario_id"])))
    return tuple(out)

def realization_run_id(spec: RealizationSpec) -> str:
    return f"RS_A5_{spec.realization_id.split('.')[-1]}"

def _surface_summary(cells):
    valid=[r for r in cells if r.get("valid") and _finite(r.get("wmarm_cs")) is not None]
    def arr(k): return np.asarray([float(r[k]) for r in valid if _finite(r.get(k)) is not None],dtype=float)
    cs,wm,kcs,bsa,pop=arr("wmarm_cs"),arr("wmarm"),arr("confidence_support_retention_ratio"),arr("bsa_coverage"),arr("population_coverage")
    return {"candidate_cell_count":len(cells),"valid_cell_count":len(valid),
            "wmarm_cs_max":float(np.max(cs)) if cs.size else None,"wmarm_cs_mean":float(np.mean(cs)) if cs.size else None,
            "diagnostic_wmarm_max":float(np.max(wm)) if wm.size else None,"diagnostic_wmarm_mean":float(np.mean(wm)) if wm.size else None,
            "k_cs_mean":float(np.mean(kcs)) if kcs.size else None,"bsa_coverage_mean":float(np.mean(bsa)) if bsa.size else None,
            "population_coverage_mean":float(np.mean(pop)) if pop.size else None,
            "zero_supported_surface":bool(cs.size and np.max(cs)<=1e-15)}

def _variance_comparison(row: Mapping[str, Any], reference: Mapping[str, Any] | None) -> dict[str, Any]:
    """RS11 Analysis 5 explicit reference-versus-realization uncertainty change."""
    realized_variance = _finite(row.get("variance"))
    reference_variance = _finite(reference.get("variance")) if reference is not None else None
    variance_change = (
        realized_variance - reference_variance
        if realized_variance is not None and reference_variance is not None else None
    )
    relative_change = (
        variance_change / reference_variance
        if variance_change is not None and reference_variance not in (None, 0.0) else None
    )
    realized_ci_low = _finite(row.get("ci_low"))
    realized_ci_high = _finite(row.get("ci_high"))
    reference_ci_low = _finite(reference.get("ci_low")) if reference is not None else None
    reference_ci_high = _finite(reference.get("ci_high")) if reference is not None else None
    return {
        "reference_variance": reference_variance,
        "variance_change": variance_change,
        "absolute_variance_change": abs(variance_change) if variance_change is not None else None,
        "relative_variance_change": relative_change,
        "variance_ratio": (
            realized_variance / reference_variance
            if realized_variance is not None and reference_variance not in (None, 0.0) else None
        ),
        "reference_ci_low": reference_ci_low,
        "reference_ci_high": reference_ci_high,
        "ci_low_change": (
            realized_ci_low - reference_ci_low
            if realized_ci_low is not None and reference_ci_low is not None else None
        ),
        "ci_high_change": (
            realized_ci_high - reference_ci_high
            if realized_ci_high is not None and reference_ci_high is not None else None
        ),
    }

def _variance_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    changes = [_finite(r.get("variance_change")) for r in rows]
    changes = [x for x in changes if x is not None]
    absolute = [_finite(r.get("absolute_variance_change")) for r in rows]
    absolute = [x for x in absolute if x is not None]
    relative = [_finite(r.get("relative_variance_change")) for r in rows]
    relative = [x for x in relative if x is not None]
    return {
        "variance_comparison_count": len(changes),
        "mean_variance_change": float(np.mean(changes)) if changes else None,
        "median_variance_change": float(np.median(changes)) if changes else None,
        "mean_absolute_variance_change": float(np.mean(absolute)) if absolute else None,
        "mean_relative_variance_change": float(np.mean(relative)) if relative else None,
    }

def _write_csv(path:Path,rows:list[dict[str,Any]]):
    path.parent.mkdir(parents=True,exist_ok=True); fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:dumps(v) if isinstance(v,(dict,list)) else clean(v) for k,v in r.items()})

def export_dependence_sensitivity(result:Mapping[str,Any],out:Path):
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'robustness_analysis5_summary.json',{k:v for k,v in result.items() if k not in ('surface_rows','yearly_rows','exclusion_rows')})
    _write_csv(out/'robustness_analysis5_realizations.csv',list(result.get('summary_rows',[])))
    _write_csv(out/'robustness_analysis5_surfaces.csv',list(result.get('surface_rows',[])))
    _write_csv(out/'robustness_analysis5_yearly.csv',list(result.get('yearly_rows',[])))
    _write_csv(out/'robustness_analysis5_exclusions.csv',list(result.get('exclusion_rows',[])))

@dataclass
class DependenceSensitivityRunner:
    data: Any
    store: Store
    persistence: RobustnessPersistence
    root: Path
    log: Any = print

    def _prepare_invariant(self, reference_config:dict, q:float, lag:int, folder:Path):
        """Prepare the dependence-invariant chronological panel once per cell."""
        d=self.data; folder.mkdir(parents=True,exist_ok=True)
        labels_path, diff_path, raw_path=folder/'labels.npy',folder/'differentials.npy',folder/'raw_weights.npy'
        if labels_path.exists() and diff_path.exists() and raw_path.exists(): return labels_path,diff_path,raw_path
        territory=d.series('territory'); labels,_=nm.exposure(territory,d.year_index,q)
        np.save(labels_path,labels,allow_pickle=False)
        support_policy=active_support_policy(reference_config)
        with mapped(diff_path,shape=(d.B,d.n)) as diffs, mapped(raw_path,shape=(d.B,d.n)) as raw:
            for b,bid in enumerate(d.ids):
                source=d.series('outcome',b)
                inc=nm.cumulative_incidence(source,d.daily_population(b),lag,reference_config['core']['incidence']['scaling_multiplier'],'EVENT_COUNTS',d.segment_index)
                base,_,donors=nm.baseline(inc,labels,reference_config['core']['baseline'],d.segment_index,d.weekday_index,d.holiday_state)
                x,_=apply_baseline_support_filter(inc-base,donors,support_policy.get('min_baseline_support'))
                local=d.series('hazard',b); w,_=nm.relevance(local,labels,d.year_index,q)
                diffs[b]=x; raw[b]=w
        return labels_path,diff_path,raw_path

    def _run_cell(self, reference_config:dict, realized:dict, spec:RealizationSpec, q:float, lag:int, cache:Path):
        d=self.data; run=realization_run_id(spec); scope=cell_scope(run,q,lag)
        prev=self.store.get('cell',run,cell_key(q,lag))
        if prev is not None:
            # A failed realization may already contain complete cells from an earlier
            # attempt. Rehydrate their yearly rows so realization-level exports and
            # diagnostics remain complete when execution resumes.
            cached_rows=[]
            for bid in d.ids:
                for year in d.years:
                    row=self.store.get('yearly',scope,f'{bid}/{year}')
                    if row is None:
                        raise RuntimeError(
                            f'Completed dependence-sensitivity cell {cell_key(q,lag)} is missing yearly row {bid}/{year}'
                        )
                    cached_rows.append(row)
            return prev,cached_rows
        labels_path,diff_path,raw_path=self._prepare_invariant(reference_config,q,lag,cache)
        dep=realized['core']['dependence']; R=realized['core']['bootstrap']['iterations']; support_policy=active_support_policy(realized)
        K=dep['maximum_acf_lag_days']; hist=np.zeros((len(d.years),K+1),dtype=np.int64)
        territory_diff=np.zeros(d.n); complete=np.ones(d.n,dtype=bool); yearly_export=[]
        self.store.delete_scope(scope)
        with mapped(labels_path) as labels, mapped(diff_path) as diffs, mapped(raw_path) as raw:
            for b,bid in enumerate(d.ids):
                x=np.asarray(diffs[b]); complete &= np.isfinite(x); territory_diff += np.where(np.isfinite(x),x,0.0)
                radius=nm.dependence(x,dep,d.segment_index)
                for iy,_year in enumerate(d.years):
                    ix=d.year_index==iy; rr=radius[ix]; rr=rr[np.isfinite(rr)].astype(int)
                    hist[iy]+=np.bincount(rr,minlength=K+1)
                    ref_support=self.store.get('support',cell_scope('REFERENCE',q,lag),f'{bid}/{_year}')
                    if ref_support is None: raise RuntimeError(f'Missing reference support row for {bid}/{_year} at q={q}, lag={lag}')
                    cloned=dict(ref_support); cloned['run']=run; self.store.put('support',scope,f'{bid}/{_year}',cloned)
            territory_diff[~complete]=np.nan; territory_radius=nm.dependence(territory_diff,dep,d.segment_index)
            for iy,year in enumerate(d.years):
                indices=np.flatnonzero(d.year_index==iy); annual_labels=np.asarray(labels[indices]); annual_segments=d.segment_index[indices]
                Dloc=nm.histogram_quantile(hist[iy],dep['annual_dependence_quantile'])
                Dtot=nm.finite_quantile(territory_radius[indices],dep['annual_dependence_quantile'])
                Bexp=nm.episode_requirement(annual_labels,dep['exposed_episode_quantile'],annual_segments)
                raw_block,block=nm.block_length_requirement(lag,Dloc,Dtot,Bexp,dep['final_block_length_multiplier'])
                reason=None
                if block is None: reason='DEPENDENCE_UNAVAILABLE'
                elif len(indices)<=block: reason='RECORD_NOT_LONGER_THAN_BLOCK'
                elif nm.max_contiguous_length(annual_segments)<=block: reason='CONTIGUOUS_RECORD_NOT_LONGER_THAN_BLOCK'
                elif not np.any(annual_labels==1) or not np.any(annual_labels==0): reason='ANNUAL_EXPOSURE_CLASS_ABSENT'
                counts_path=cache/f'draw_counts_{spec.realization_id}_{year}.npy'
                rc_paths=[]
                try:
                    with ExitStack() as stack:
                        counts=None; redraws=0
                        if reason is None:
                            counts=stack.enter_context(mapped(counts_path,shape=(R,len(indices)),dtype='uint16'))
                            rng=rng_for_identity(realized['seed'],namespace='DISK_DEV_BOOTSTRAP_V1',identity={'run':'REFERENCE','q':float(q),'lag':lag,'year':year})
                            redraws=nm.draw_counts(annual_labels,block,rng,counts,annual_segments)
                        for b,bid in enumerate(d.ids):
                            x=np.asarray(diffs[b,indices]); w=np.asarray(raw[b,indices])
                            original=float(nm.weighted_delta_batch(x,w,annual_labels,np.ones((1,len(indices)),dtype=np.uint16))[0])
                            ref=self.store.get('yearly',cell_scope('REFERENCE',q,lag),f'{bid}/{year}')
                            ref_rc=None if ref is None else ref.get('rc')
                            if ref_rc is not None and np.isfinite(original) and abs(float(ref_rc)-original)>1e-12:
                                raise RuntimeError(f'Dependence-only realization changed full-sample RC for {bid}/{year}: {original} vs {ref_rc}')
                            draw_path=cache/f'rc_{spec.realization_id}_{year}_{b}.npy'
                            rc_paths.append(draw_path)
                            with mapped(draw_path,shape=(R,)) as draws:
                                draws[:]=np.nan
                                if counts is not None and np.isfinite(original):
                                    bs=realized['runtime']['bootstrap_batch_size']
                                    for r in range(0,R,bs): draws[r:r+bs]=nm.weighted_delta_batch(x,w,annual_labels,counts[r:r+bs])
                                summary=nm.bootstrap_summary(original,draws,R,realized['core']['bootstrap']['maximum_invalid_realization_proportion'])
                            support=self.store.get('support',scope,f'{bid}/{year}')
                            failures=list(support.get('support_failures_prebootstrap') or [])
                            failures.extend(evaluate_postbootstrap_support(summary,support_policy))
                            if reason is not None: summary.update(eligible=False,reason=reason)
                            elif summary.get('eligible') and failures: summary.update(eligible=False,reason=failures[0])
                            summary['support_failures']=failures; summary['support_eligible']=not failures
                            row={**support,**summary,'D_local':Dloc,'D_territory':Dtot,'exposure_block_requirement':Bexp,
                                 'raw_block_length':raw_block,'block_length_multiplier':dep['final_block_length_multiplier'],'block_length':block,
                                 'global_redraws':redraws,'block_reason':reason,'outcome_semantics':'EVENT_COUNTS',
                                 'reference_rc':ref_rc,'rc_invariance_abs_difference':abs(original-float(ref_rc)) if ref_rc is not None and np.isfinite(original) else None}
                            row.update(_variance_comparison(row, ref))
                            self.store.put('yearly',scope,f'{bid}/{year}',row)
                            yearly_export.append(row)
                finally:
                    # Windows does not permit unlinking a file while an np.memmap still owns it.
                    # ExitStack closes the shared yearly counts mapping before any scratch cleanup.
                    for path in rc_paths:
                        path.unlink(missing_ok=True)
                    counts_path.unlink(missing_ok=True)
        engine=FixedParametersEngine(d,realized,self.store,self.root,self.log)
        cell=engine.pool_cell(run,q,lag,excluded_year=None,save=True)
        cell.update(run=run,q=q,lag=lag,outcome_semantics='EVENT_COUNTS')
        self.store.put('cell',run,cell_key(q,lag),cell); self.store.commit()
        return cell,yearly_export

    def run(self, reference_config:dict, plan) -> dict[str,Any]:
        specs=[r for r in plan.realizations if r.analysis_id==ANALYSIS_ID]
        summaries=[]; surfaces=[]; yearly=[]; exclusions=[]
        cache_root=self.root/'scratch'/'rs_analysis5_invariant'; cache_root.mkdir(parents=True,exist_ok=True)
        for spec in specs:
            cp=self.persistence.checkpoint(spec.realization_id,CHECKPOINT_STAGE,'REALIZATION','result')
            if cp and cp.get('state')==CheckpointState.COMPLETED.value:
                stored=self.persistence.analysis_output(spec.realization_id,ANALYSIS_ID,'SUMMARY','REALIZATION','result')
                if stored is not None:
                    # 2.3.4 output completion: enrich an already-completed
                    # 2.3.3 realization from persisted yearly/reference rows,
                    # without rerunning dependence calibration or bootstrap.
                    hydrated=[]
                    persisted_yearly=list(self.persistence.analysis_outputs(spec.realization_id,ANALYSIS_ID,'YEARLY','BSA_YEAR'))
                    for item in persisted_yearly:
                        payload=dict(item['payload'])
                        ref=self.store.get(
                            'yearly', cell_scope('REFERENCE',float(payload['q']),int(payload['lag'])),
                            f"{payload['bsa_id']}/{int(payload['year'])}"
                        )
                        payload.update(_variance_comparison(payload,ref))
                        key=f"{float(payload['q']):g}/{int(payload['lag']):04d}/{payload['bsa_id']}/{int(payload['year'])}"
                        self.persistence.put_analysis_output(
                            spec.realization_id,ANALYSIS_ID,'YEARLY','BSA_YEAR',key,payload
                        )
                        hydrated.append(payload)
                    stored={**stored,'schema_version':SCHEMA_VERSION,**_variance_summary(hydrated)}
                    self.persistence.put_analysis_output(
                        spec.realization_id,ANALYSIS_ID,'SUMMARY','REALIZATION','result',stored
                    )
                    self.persistence.put_checkpoint(
                        spec.realization_id,CHECKPOINT_STAGE,'REALIZATION','result',CheckpointState.COMPLETED,payload=stored
                    )
                    summaries.append(stored)
                    surfaces.extend(r['payload'] for r in self.persistence.analysis_outputs(spec.realization_id,ANALYSIS_ID,'SURFACE','CELL'))
                    yearly.extend(hydrated)
                    exclusions.extend(r['payload'] for r in self.persistence.analysis_outputs(spec.realization_id,ANALYSIS_ID,'EXCLUSION','COUNT'))
                    continue
            self.persistence.set_realization_state(spec.realization_id,RealizationState.RUNNING)
            self.persistence.put_checkpoint(spec.realization_id,CHECKPOINT_STAGE,'REALIZATION','result',CheckpointState.RUNNING)
            try:
                realized=materialize_realization(reference_config,spec); validate(realized)
                self.log(f'RS Analysis 5: {spec.scenario_id} step={spec.step_index} dependence={realized["core"]["dependence"]}')
                cells=[]; yr=[]
                for q in realized['thresholds']:
                    for lag in realized['lags']:
                        cache=cache_root/hashlib.sha256(cell_scope('REFERENCE',q,lag).encode()).hexdigest()[:24]
                        cell,rows=self._run_cell(reference_config,realized,spec,q,lag,cache); cells.append(cell); yr.extend(rows)
                inv=[_finite(r.get('rc_invariance_abs_difference')) for r in yr]; inv=[x for x in inv if x is not None]
                blocks=[_finite(r.get('block_length')) for r in yr]; blocks=[x for x in blocks if x is not None]
                rawblocks=[_finite(r.get('raw_block_length')) for r in yr]; rawblocks=[x for x in rawblocks if x is not None]
                row={'schema_version':SCHEMA_VERSION,'analysis_id':ANALYSIS_ID,'realization_id':spec.realization_id,'scenario_id':spec.scenario_id,
                     'step_index':spec.step_index,'trajectory_coordinate':spec.trajectory_coordinate,'run_id':realization_run_id(spec),
                     'dependence_policy':dict(realized['core']['dependence']),'invalidates_from':spec.invalidates_from.name,
                     'full_sample_rc_max_abs_difference':max(inv) if inv else None,'full_sample_rc_invariant':bool(inv and max(inv)<=1e-12),
                     'mean_raw_block_length':float(np.mean(rawblocks)) if rawblocks else None,'mean_adopted_block_length':float(np.mean(blocks)) if blocks else None,
                     **_variance_summary(yr),**_surface_summary(cells)}
                self.persistence.clear_analysis_outputs(spec.realization_id,ANALYSIS_ID)
                self.persistence.put_analysis_output(spec.realization_id,ANALYSIS_ID,'SUMMARY','REALIZATION','result',row)
                for cell in cells:
                    payload={'analysis_id':ANALYSIS_ID,'realization_id':spec.realization_id,'scenario_id':spec.scenario_id,'step_index':spec.step_index,
                             'trajectory_coordinate':spec.trajectory_coordinate,'dependence_policy':dict(realized['core']['dependence']),**cell}
                    key=f"q{float(cell['q']):g}_lag{int(cell['lag']):04d}"; self.persistence.put_analysis_output(spec.realization_id,ANALYSIS_ID,'SURFACE','CELL',key,payload); surfaces.append(payload)
                counts={}
                for r in yr:
                    payload={'analysis_id':ANALYSIS_ID,'realization_id':spec.realization_id,'scenario_id':spec.scenario_id,'step_index':spec.step_index,**r}
                    key=f"{r['q']:g}/{int(r['lag']):04d}/{r['bsa_id']}/{r['year']}"; self.persistence.put_analysis_output(spec.realization_id,ANALYSIS_ID,'YEARLY','BSA_YEAR',key,payload); yearly.append(payload)
                    reason='ELIGIBLE' if r.get('eligible') else str(r.get('reason') or 'INELIGIBLE'); counts[reason]=counts.get(reason,0)+1
                for i,(reason,count) in enumerate(sorted(counts.items())):
                    payload={'analysis_id':ANALYSIS_ID,'realization_id':spec.realization_id,'scenario_id':spec.scenario_id,'step_index':spec.step_index,'level':'YEARLY','reason':reason,'count':count}
                    self.persistence.put_analysis_output(spec.realization_id,ANALYSIS_ID,'EXCLUSION','COUNT',f'{i:04d}',payload); exclusions.append(payload)
                self.persistence.put_checkpoint(spec.realization_id,CHECKPOINT_STAGE,'REALIZATION','result',CheckpointState.COMPLETED,payload=row)
                self.persistence.set_realization_state(spec.realization_id,RealizationState.COMPLETED); summaries.append(row)
            except BaseException as exc:
                self.persistence.put_checkpoint(spec.realization_id,CHECKPOINT_STAGE,'REALIZATION','result',CheckpointState.FAILED,error=f'{type(exc).__name__}: {exc}')
                self.persistence.set_realization_state(spec.realization_id,RealizationState.FAILED,error=f'{type(exc).__name__}: {exc}'); raise
        shutil.rmtree(cache_root,ignore_errors=True)
        result={'schema_version':SCHEMA_VERSION,'analysis_id':ANALYSIS_ID,'status':'COMPLETED' if specs else 'NOT_REQUESTED','realization_count':len(specs),
                'reference_dependence_policy':dict(reference_config['core']['dependence']),'summary_rows':summaries,'surface_rows':surfaces,'yearly_rows':yearly,'exclusion_rows':exclusions,
                'scope_note':'Step 4 reruns dependence calibration, synchronized bootstrap, pooling and territorial surfaces; Step 7 provides fixed-parameter comparison and Step 8 optionally performs complete realization-specific reoptimization.'}
        export_dependence_sensitivity(result,self.root/'results'); return result
