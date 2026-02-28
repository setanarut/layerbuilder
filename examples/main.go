package main

import (
	// Import the layerbuilder package with a short alias.
	lb "layerbuilder"
	// Import utility helpers for image I/O and palette operations.
	"layerbuilder/utils"
)

func main() {
	// Read the source image.
	img := utils.ReadImage("../_test_files/parrot.png")
	// Extract a 7-color palette using the dominant-color method.
	palette := utils.ExtractPalette(img, 7, utils.PaletteMethodDominantColor)
	// Sort palette colors from dark to bright for better results.
	utils.SortPaletteByBrightness(palette)

	// Create a new builder with the image and palette.
	builder := lb.NewLayerBuilder(img, palette)
	// Build internal layers using options derived from image size.
	builder.Build(lb.OptionsFromSize(img.Bounds().Size()))
	// Reconstruct an image from grayscale alpha layers.
	recon := builder.Reconstruct(builder.GrayLayers())

	// Save reconstructed image and debug outputs.
	utils.SaveImage(recon, "../_test_files/output/recon.png")
	utils.SavePalette(palette, 64, "../_test_files/output/palette.png")
	utils.SaveRgbaImages(builder.RGBALayers(), "../_test_files/output/")
}
