package main

import (
	lb "layerbuilder"
	"layerbuilder/utils"
)

func main() {
	img := utils.ReadImage("../_test_files/snooker.png")
	palette := utils.ExtractPalette(img, 7, utils.PaletteMethodDominantColor)
	utils.SortPaletteByBrightness(palette)

	builder := lb.NewLayerBuilder(img, palette)
	builder.Build(lb.OptionsFromSize(img.Bounds().Size()))
	recon := builder.Reconstruct(builder.GrayLayers())

	utils.SaveImage(recon, "../_test_files/output/recon.png")
	utils.SavePalette(palette, 64, "../_test_files/output/palette.png")
	utils.SaveRgbaImages(builder.RGBALayers(), "../_test_files/output/")

}
