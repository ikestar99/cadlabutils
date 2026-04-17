#@ String inputPath
#@ String outputPath
#@ int radius
#@ String background

setBatchMode(true)
open(inputPath)
origTitle = getTitle()
origType = bitDepth()
run("32-bit")
run("Subtract Background...", "rolling=" + radius + " " + background + " create")
bgTitle = getTitle()
saveAs("Tiff", outputPath)
close()
close(origTitle)
