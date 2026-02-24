package utils

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"
	"slices"
	"strconv"

	"github.com/cenkalti/dominantcolor"
	"github.com/lucasb-eyer/go-colorful"
	"github.com/muesli/clusters"
	"github.com/muesli/kmeans"
)

type PaletteMethod int

const (
	PaletteMethodDominantColor PaletteMethod = iota
	PaletteMethodKMeans
)

type weightedColor struct {
	Col    colorful.Color
	Weight float64
}

// SortPaletteByBrightness orders colors from darkest to brightest.
// The first palette entry becomes the darkest color (background).
func SortPaletteByBrightness(palette []colorful.Color) {
	slices.SortFunc(palette, func(a, b colorful.Color) int {
		ri, gi, bi := a.LinearRgb()
		rj, gj, bj := b.LinearRgb()
		yi := 0.2126*ri + 0.7152*gi + 0.0722*bi
		yj := 0.2126*rj + 0.7152*gj + 0.0722*bj
		if yi < yj {
			return -1
		}
		if yi > yj {
			return 1
		}
		return 0
	})
}

func (m PaletteMethod) String() string {
	switch m {
	case PaletteMethodKMeans:
		return "kmeans"
	default:
		return "dominantcolor"
	}
}

func ExtractDominantPalette(img image.Image, k int) []colorful.Color {
	if k <= 0 {
		return nil
	}

	nCandidates := max(24, k*8)
	candidates := dominantcolor.FindWeight(img, nCandidates)
	if len(candidates) == 0 {
		// Last resort: avoid empty palette that would break downstream solves.
		candidates = append(candidates, dominantcolor.Color{
			RGBA:   color.RGBA{R: 128, G: 128, B: 128, A: 255},
			Weight: 1.0,
		})
	}

	weighted := make([]weightedColor, 0, len(candidates))
	for _, c := range candidates {
		col, _ := colorful.MakeColor(c.RGBA)
		w := c.Weight
		if w <= 0 {
			w = 1e-6
		}
		weighted = append(weighted, weightedColor{Col: col.Clamped(), Weight: w})
	}
	return SelectDiverseWeightedColors(weighted, k)
}

func SelectDiverseWeightedColors(cands []weightedColor, k int) []colorful.Color {
	if k <= 0 || len(cands) == 0 {
		return nil
	}
	type item struct {
		col colorful.Color
		lab [3]float64
		w   float64
	}
	items := make([]item, 0, len(cands))
	maxW := 0.0
	for _, c := range cands {
		col := c.Col.Clamped()
		l, a, b := col.Lab()
		w := c.Weight
		if w <= 0 {
			w = 1e-6
		}
		if w > maxW {
			maxW = w
		}
		items = append(items, item{
			col: col,
			lab: [3]float64{l, a, b},
			w:   w,
		})
	}
	if len(items) == 0 {
		return nil
	}
	if k > len(items) {
		k = len(items)
	}
	if maxW <= 0 {
		maxW = 1.0
	}

	selectedIdx := make([]int, 0, k)
	selected := make([]bool, len(items))

	// Seed with strongest color to stay close to dominant tones.
	bestSeed := 0
	bestSeedW := items[0].w
	for i := 1; i < len(items); i++ {
		if items[i].w > bestSeedW {
			bestSeedW = items[i].w
			bestSeed = i
		}
	}
	selectedIdx = append(selectedIdx, bestSeed)
	selected[bestSeed] = true

	for len(selectedIdx) < k {
		bestIdx := -1
		bestScore := -1.0
		for i := range items {
			if selected[i] {
				continue
			}
			minD2 := math.MaxFloat64
			for _, s := range selectedIdx {
				d0 := items[i].lab[0] - items[s].lab[0]
				d1 := items[i].lab[1] - items[s].lab[1]
				d2 := items[i].lab[2] - items[s].lab[2]
				d2v := d0*d0 + d1*d1 + d2*d2
				if d2v < minD2 {
					minD2 = d2v
				}
			}
			normW := items[i].w / maxW
			score := math.Sqrt(minD2) * (0.55 + 0.45*math.Sqrt(normW))
			if score > bestScore {
				bestScore = score
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		selected[bestIdx] = true
		selectedIdx = append(selectedIdx, bestIdx)
	}

	out := make([]colorful.Color, 0, len(selectedIdx))
	for _, idx := range selectedIdx {
		out = append(out, items[idx].col)
	}
	return out
}

func ExtractKMeansPalette(img image.Image, k int) []colorful.Color {
	if k <= 0 {
		return nil
	}

	b := img.Bounds()
	width, height := b.Dx(), b.Dy()
	if width == 0 || height == 0 {
		return nil
	}

	// Subsample to keep kmeans tractable on large images.
	maxSamples := 12000
	step := 1
	if width*height > maxSamples {
		step = int(math.Sqrt(float64(width*height)/float64(maxSamples))) + 1
	}

	dataset := make(clusters.Observations, 0, min(width*height, maxSamples))
	for y := b.Min.Y; y < b.Max.Y; y += step {
		for x := b.Min.X; x < b.Max.X; x += step {
			r16, g16, b16, a16 := img.At(x, y).RGBA()
			if a16 == 0 {
				continue
			}
			dataset = append(dataset, clusters.Coordinates{
				float64(r16) / 65535.0,
				float64(g16) / 65535.0,
				float64(b16) / 65535.0,
			})
		}
	}
	if len(dataset) == 0 {
		return nil
	}

	workK := min(max(k*4, k+2), len(dataset))
	if workK <= 0 {
		return nil
	}
	km := kmeans.New()
	cc, err := km.Partition(dataset, workK)
	if err != nil || len(cc) == 0 {
		return nil
	}

	// Sort by cluster population so dominant colors come first.
	slices.SortFunc(cc, func(a, b clusters.Cluster) int {
		na := len(a.Observations)
		nb := len(b.Observations)
		if na > nb {
			return -1
		}
		if na < nb {
			return 1
		}
		return 0
	})

	weighted := make([]weightedColor, 0, len(cc))
	for _, c := range cc {
		center := c.Center
		if len(center) < 3 {
			continue
		}
		col := colorful.Color{
			R: center[0],
			G: center[1],
			B: center[2],
		}.Clamped()
		w := float64(len(c.Observations))
		if w <= 0 {
			w = 1e-6
		}
		weighted = append(weighted, weightedColor{Col: col, Weight: w})
	}
	return SelectDiverseWeightedColors(weighted, k)
}

func ExtractPalette(img image.Image, k int, method PaletteMethod) []colorful.Color {
	switch method {
	case PaletteMethodKMeans:
		p := ExtractKMeansPalette(img, k)
		if len(p) != 0 {
			return p
		}
		log.Println("palette warning: kmeans returned empty palette, falling back to dominantcolor")
		return ExtractDominantPalette(img, k)
	default:
		return ExtractDominantPalette(img, k)
	}
}

func ReadImage(path string) image.Image {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		panic(err)
	}
	return img
}

func SaveRgbaImages(images []*image.NRGBA, dir string) error {
	for i := range images {
		if err := SaveImage(images[i], dir+"rgba_0"+strconv.Itoa(i)+".png"); err != nil {
			return err
		}
	}
	return nil
}
func SaveGrayImages(images []*image.Gray, dir string) error {
	for i := range images {
		if err := SaveImage(images[i], dir+"gray_0"+strconv.Itoa(i)+".png"); err != nil {
			return err
		}
	}
	return nil
}

func SaveImage(img image.Image, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func SavePalette(palette []colorful.Color, tileSize int, filename string) error {
	if len(palette) == 0 {
		return fmt.Errorf("empty palette")
	}
	if tileSize <= 0 {
		tileSize = 64
	}

	w := tileSize * len(palette)
	h := tileSize
	img := image.NewRGBA(image.Rect(0, 0, w, h))

	for i, c := range palette {
		r := uint8(max(0, min(255, c.R*255)))
		g := uint8(max(0, min(255, c.G*255)))
		b := uint8(max(0, min(255, c.B*255)))
		x0 := i * tileSize
		x1 := x0 + tileSize
		for y := range h {
			for x := x0; x < x1; x++ {
				img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
			}
		}
	}

	return SaveImage(img, filename)
}
