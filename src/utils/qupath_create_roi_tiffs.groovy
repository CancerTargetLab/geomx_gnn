// Write the region of the image corresponding to the currently-selected object
def project = getProject()

for (entry in project.getImageList()) {
    // Run detection for the all objects
    def imageData = entry.readImageData()
    def server = imageData.getServer()
    def hierarchy = imageData.getHierarchy()
    def pathObjects = hierarchy.getAnnotationObjects()
    if (pathObjects.isEmpty()) {
        QP.getLogger().error("No parent objects are selected!")
        return
    }
    for (roi in pathObjects) {
        //println roi.getName()
        //println describe(roi)
        def requestROI = RegionRequest.createInstance(server.getPath(), 1, roi.getROI())
        writeImageRegion(server, requestROI, '/path/to/dir/'+roi.getName()+'-'+entry.getImageName())
        }
}
println('Done!')