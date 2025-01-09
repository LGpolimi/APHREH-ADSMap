import pandas as pd
import numpy as np
import sys, os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'D:\OSGeo4W64\apps\Qt5\plugins'
os.environ['QT_PLUGIN_PATH'] = r'%QT_PLUGIN_PATH%;D:\OSGeo4W64\apps\Qt5\plugins;D:\OSGeo4W64\apps\qgis\qtplugins;D:\OSGeo4W64\apps\qgis\plugins'
os.environ['PATH'] += r';D:\OSGeo4W64\apps\qgis\bin;D:\OSGeo4W64\apps\Qt5\bin;D:\OSGeo4W64\bin'
os.environ['PATH'] += r';D:\OSGeo4W64\apps\grass\grass76\lib'
os.environ['QGIS_PREFIX'] = 'D:/OSGeo4W64/apps/qgis'
os.environ['QGIS_PREFIX_PATH'] = 'D:/OSGeo4W64/apps/qgis'
os.environ['GDAL_DATA'] = 'D:/OSGeo4W64/share/gdal'
os.environ['MAX_CACHE_LAYERS'] = '100000'
sys.path.append('D:\\OSGeo4W64\\apps\\qgis-ltr\\python\\plugins')
sys.path.append('D:\\OSGeo4W64\\apps\\qgis-ltr')
sys.path.append('D:\\OSGeo4W64\\apps\\qgis-ltr\\grass')
sys.path.append('D:\\OSGeo4W64\\apps\\grass\\grass76')
gisbase = os.environ['GISBASE'] = 'D:\\OSGeo4W64\\apps\\grass\\grass76'

environment_variables = os.environ.copy()
from qgis.core import QgsApplication, QgsVectorLayer, QgsVectorFileWriter
from qgis.analysis import QgsNativeAlgorithms
QgsApplication.setPrefixPath("D:/OSGeo4W64/apps/qgis", True)
qgs = QgsApplication([], False)
qgs.initQgis()
try:
    import processing
    from processing.core.Processing import Processing
except ImportError as e:
    print("Error importing processing module:", e)
    sys.exit(1)
Processing.initialize()
qgs.processingRegistry().addProvider(QgsNativeAlgorithms())


rootpath = 'D:\\PythonDEV\\QGIS_support\\'
geoarea_highgr = 'MIL0B' # Higher granularity layer: to be aggregated
geoarea_lowgr = 'MIL2A' # Lower granularity layer: boundareis to be maintained
pop_field = 'POP_2021'
highgr_uid = 'MIL0B_IDcu' # Must be a numerical field
lowgr_uid = 'MIL2A_IDis' # Must be a numerical field
saveout = 1

high_gr_path = os.path.join(rootpath, geoarea_highgr + '.shp')
low_gr_path = os.path.join(rootpath, geoarea_lowgr + '.shp')
high_gr_areas_layer = QgsVectorLayer(high_gr_path , geoarea_highgr, "ogr")
low_gr_areas_layer = QgsVectorLayer(low_gr_path , geoarea_lowgr, "ogr")
if not high_gr_areas_layer.isValid() or not low_gr_areas_layer.isValid():
    print("ERROR in layers loading!")
    if not high_gr_areas_layer.isValid():
        print(f"Failed to load high granularity layer: {rootpath+geoarea_highgr+'.shp'}")
    if not low_gr_areas_layer.isValid():
        print(f"Failed to load low granularity layer: {rootpath+geoarea_lowgr+'.shp'}")
    sys.exit(1)

# IF NOT DONE ALREADY, RUN THE INTERSECTION BETWEEN THE TWO LAYERS TO ESTABLISH BELONGINGS
intersection_result = processing.run("qgis:intersection", {
    'INPUT': high_gr_areas_layer,
    'OVERLAY': low_gr_areas_layer,
    'OUTPUT': 'memory:'
})
intersection_layer = intersection_result['OUTPUT']
crs = high_gr_areas_layer.crs()
intersection_layer.setCrs(crs)
output_path = rootpath + 'intersection_output.shp'
error = QgsVectorFileWriter.writeAsVectorFormat(intersection_layer, output_path, "utf-8", crs, "ESRI Shapefile")
if error == QgsVectorFileWriter.NoError:
    print("Intersection shapefile saved successfully")
else:
    print("Error saving intersection shapefile!")

br = 1

qgs.exitQgis()

br = 1