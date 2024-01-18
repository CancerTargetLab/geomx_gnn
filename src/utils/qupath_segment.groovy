/**
 * This script provides a general template for cell detection using StarDist in QuPath.
 * This example assumes you have fluorescence image, which has a channel called 'DAPI' 
 * showing nuclei.
 * 
 * If you use this in published work, please remember to cite *both*:
 *  - the original StarDist paper (https://doi.org/10.48550/arXiv.1806.03535)
 *  - the original QuPath paper (https://doi.org/10.1038/s41598-017-17204-5)
 *  
 * There are lots of options to customize the detection - this script shows some 
 * of the main ones. Check out other scripts and the QuPath docs for more info.
 */

import qupath.ext.stardist.StarDist2D
import qupath.lib.scripting.QP


// IMPORTANT! Replace this with the path to your StarDist model
// that takes a single channel as input (e.g. dsb2018_heavy_augment.pb)
// You can find some at https://github.com/qupath/models
// (Check credit & reuse info before downloading)
def modelPath = "/proj/berzelius-2023-241/users/x_mahei/QPath/dsb2018_heavy_augment.pb"

//REQUIRED CONVERSION FOR GPU PROCESSING
def dnn = DnnTools.builder(modelPath).build();

// Customize how the StarDist detection should be applied
// Here some reasonable default options are specified
def stardist = StarDist2D
    .builder(dnn)
    .channels('FITC/525nm')            // Extract channel called 'DAPI'
    .normalizePercentiles(1, 99) // Percentile normalization
    .threshold(0.5)              // Probability (detection) threshold
    .pixelSize(0.5)              // Resolution for detection
    .cellExpansion(2)            // Expand nuclei to approximate cell boundaries
    .measureShape()              // Add shape measurements
    .measureIntensity()          // Add cell measurements (in all compartments)
    .build()

def imageData = getCurrentImageData()
def pathObjects = QP.getAnnotationObjects()
if (pathObjects.isEmpty()) {
    QP.getLogger().error("No parent objects are selected!")
    return
}
stardist.detectObjects(imageData, pathObjects)

//hierarchy.resolveHierarchy()
//entry.saveImageData(imageData)
//println entry.getImageName() + ' DONE'

stardist.close() // This can help clean up & regain memory
//CLOSE DNN TO FREE UP VRAM
dnn.getPredictionFunction().net.close()

println('Done!')

//./qupath/build/dist/QuPath/bin/QuPath script -p=1C-54/project.qpproj segment.groovy