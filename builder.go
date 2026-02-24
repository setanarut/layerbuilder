package layerbuilder

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/lucasb-eyer/go-colorful"
	"gonum.org/v1/gonum/mat"
)

type Options struct {
	// Number of SLIC regions.
	// Image-size dependent: larger images usually need higher values.
	// Ideal start: ~area/(40^2) (roughly 300-1200 for common inputs).
	// Too low => larger regions => blockier alpha transitions.
	NumSuperpixels int
	// Neighbor count for superpixel manifold graph W.
	// Ideal start: 24-40.
	// Too low can over-localize layers; too high can over-smooth boundaries.
	LLENeighbors int
	// Per-pixel interpolation neighbors from superpixels. Most direct blockiness control:
	// lower values (especially 1-4) can create visible region boundaries.
	// Ideal start: 12-20.
	PixelNeighbors int
	// Manifold smoothness weight in solveFullSystem.
	// Ideal start: 0.18-0.30. Lower => weaker cross-region smoothing.
	Lm float64
	// Reconstruction weight in solveFullSystem.
	// Ideal start: ~1.0 (0.8-1.2). Lower => weaker color fitting, more seed-like alphas.
	Lr float64
	// Alpha magnitude regularization.
	// Ideal start: 0.01-0.03. Higher pushes alphas toward 0 and can produce sparse blobs.
	Lu float64
}

func DefaultOptions() Options {
	return Options{
		NumSuperpixels: 500,
		LLENeighbors:   30,
		PixelNeighbors: 16,
		Lm:             0.22,
		Lr:             1.0,
		Lu:             0.015,
	}
}

func OptionsFromSize(size image.Point) Options {
	if size.X <= 0 || size.Y <= 0 {
		return DefaultOptions()
	}
	pixels := size.X * size.Y
	targetStep := 40.0
	if pixels <= 512*512 {
		targetStep = 32.0
	} else if pixels > 1920*1080 {
		targetStep = 48.0
	}
	nsp := int(float64(pixels) / (targetStep * targetStep))
	nsp = max(150, min(2000, nsp))

	opt := DefaultOptions()
	opt.NumSuperpixels = nsp
	return opt
}

type LayerBuilder struct {
	InputImage      image.Image
	Rgb             rgb32
	Lab             lab32
	Clusters        labelImage
	SuperPixels     []superpixel
	LLEWeightMatrix *mat.Dense
	L               *mat.Dense
	PixelWeights    []float64
	Palette         []colorful.Color
}

func NewLayerBuilder(input image.Image, palette []colorful.Color) *LayerBuilder {
	return &LayerBuilder{
		InputImage: input,
		Palette:    palette,
	}
}

func (lb *LayerBuilder) Reconstruct(grayLayers []*image.Gray) *image.RGBA {
	w, h := lb.Rgb.W, lb.Rgb.H
	recon := image.NewRGBA(image.Rect(0, 0, w, h))
	numChannels := len(grayLayers)
	if numChannels == 0 || len(lb.Palette) == 0 {
		return recon
	}
	for y := range h {
		for x := range w {
			// Start from opaque background color and alpha-composite foregrounds bottom -> top.
			outR := lb.Palette[0].R
			outG := lb.Palette[0].G
			outB := lb.Palette[0].B
			for ch := 1; ch < numChannels; ch++ { // bottom -> top, palette order
				a := float64(grayLayers[ch].GrayAt(x, y).Y) / 255.0
				if a == 0 {
					continue
				}
				oneMinusA := 1 - a
				outR = a*lb.Palette[ch].R + oneMinusA*outR
				outG = a*lb.Palette[ch].G + oneMinusA*outG
				outB = a*lb.Palette[ch].B + oneMinusA*outB
			}
			recon.SetRGBA(x, y, color.RGBA{
				uint8(max(0, min(255, outR*255))),
				uint8(max(0, min(255, outG*255))),
				uint8(max(0, min(255, outB*255))),
				255, // Opaque result over fixed background.
			})
		}
	}
	return recon
}

func (lb *LayerBuilder) RGBALayers() []*image.NRGBA {
	numChannels := len(lb.Palette)
	if numChannels == 0 || lb.Rgb.W == 0 || lb.Rgb.H == 0 || len(lb.PixelWeights) == 0 {
		return nil
	}
	out := make([]*image.NRGBA, numChannels)
	w, h := lb.Rgb.W, lb.Rgb.H
	for ch := range numChannels {
		layer := image.NewNRGBA(image.Rect(0, 0, w, h))
		cr := uint8(max(0, min(255, lb.Palette[ch].R*255)))
		cg := uint8(max(0, min(255, lb.Palette[ch].G*255)))
		cb := uint8(max(0, min(255, lb.Palette[ch].B*255)))
		for y := range h {
			for x := range w {
				base := (y*w + x) * numChannels
				a := lb.PixelWeights[base+ch]
				if a < 0 {
					a = 0
				}
				if a > 1 {
					a = 1
				}
				layer.SetNRGBA(x, y, color.NRGBA{R: cr, G: cg, B: cb, A: uint8(a * 255)})
			}
		}
		out[ch] = layer
	}
	return out
}

func (lb *LayerBuilder) GrayLayers() []*image.Gray {
	numChannels := len(lb.Palette)
	if numChannels == 0 || lb.Rgb.W == 0 || lb.Rgb.H == 0 || len(lb.PixelWeights) == 0 {
		return nil
	}
	out := make([]*image.Gray, numChannels)
	w, h := lb.Rgb.W, lb.Rgb.H
	for ch := range numChannels {
		layer := image.NewGray(image.Rect(0, 0, w, h))
		for y := range h {
			for x := range w {
				base := (y*w + x) * numChannels
				a := lb.PixelWeights[base+ch]
				if a < 0 {
					a = 0
				}
				if a > 1 {
					a = 1
				}
				layer.SetGray(x, y, color.Gray{Y: uint8(a * 255)})
			}
		}
		out[ch] = layer
	}
	return out
}

func (lb *LayerBuilder) Build(opt Options) {
	lb.makeRGB32Image()
	lb.makeLab32ImageFromRGB32()
	lb.slic(opt.NumSuperpixels)
	lb.computeSuperpixels()
	lb.buildLLEWeightMatrix(opt.LLENeighbors)
	lb.solveFullSystem(opt.Lm, opt.Lr, opt.Lu)
	lb.computePerPixelLLEWeights(opt.PixelNeighbors)
}

type rgb32 struct {
	W, H int
	Pix  []float32 // Interleaved RGB in [0,255], len = W*H*3
}

type lab32 struct {
	W, H int
	Pix  []float32 // Interleaved LAB, len = W*H*3
}

type labelImage struct {
	W, H   int
	Labels []int // len = W*H
}

type superpixel struct {
	CX, CY float32
	lab    colorful.Color // L,a,b stored in R,G,B channels
	rgb    colorful.Color // normalized RGB in [0,1]
}

type accumulator struct {
	l, a, b, cx, cy, r, g, bl float64
	count                     int
}

func pixOffset(w, x, y int) int {
	return (y*w + x) * 3
}

func labelOffset(w, x, y int) int {
	return y*w + x
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func minDistanceToOutsideCellsSquared(px, py float64, minGX, maxGX, minGY, maxGY, gridW, gridH int, cellSize float64) float64 {
	minDist := math.Inf(1)

	if minGX > 0 {
		minDist = min(minDist, px-float64(minGX)*cellSize)
	}
	if maxGX < gridW-1 {
		minDist = min(minDist, float64(maxGX+1)*cellSize-px)
	}
	if minGY > 0 {
		minDist = min(minDist, py-float64(minGY)*cellSize)
	}
	if maxGY < gridH-1 {
		minDist = min(minDist, float64(maxGY+1)*cellSize-py)
	}

	if math.IsInf(minDist, 1) {
		return math.Inf(1)
	}

	minDist = max(minDist, 0)
	return minDist * minDist
}
func (lb *LayerBuilder) makeRGB32Image() {
	bounds := lb.InputImage.Bounds()
	h := bounds.Dy()
	w := bounds.Dx()
	lb.Rgb = rgb32{
		W:   w,
		H:   h,
		Pix: make([]float32, h*w*3),
	}
	for y := range h {
		for x := range w {
			r, g, b, _ := lb.InputImage.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			off := pixOffset(w, x, y)
			lb.Rgb.Pix[off] = float32(r >> 8)
			lb.Rgb.Pix[off+1] = float32(g >> 8)
			lb.Rgb.Pix[off+2] = float32(b >> 8)
		}
	}
}

// ============ RGB → LAB ============

func (lb *LayerBuilder) makeLab32ImageFromRGB32() {
	h := lb.Rgb.H
	w := lb.Rgb.W
	lb.Lab = lab32{
		W:   w,
		H:   h,
		Pix: make([]float32, h*w*3),
	}
	for y := range h {
		for x := range w {
			off := pixOffset(w, x, y)
			c := colorful.Color{
				R: float64(lb.Rgb.Pix[off]) / 255.0,
				G: float64(lb.Rgb.Pix[off+1]) / 255.0,
				B: float64(lb.Rgb.Pix[off+2]) / 255.0,
			}
			l, a, b := c.Lab()
			lb.Lab.Pix[off] = float32(l)
			lb.Lab.Pix[off+1] = float32(a)
			lb.Lab.Pix[off+2] = float32(b)
		}
	}
}

// ============ SLIC ============

func (lb *LayerBuilder) slic(numSuperpixels int) {
	lab := lb.Lab
	h := lab.H
	w := lab.W
	if numSuperpixels <= 0 {
		numSuperpixels = 1
	}
	step := max(int(math.Sqrt(float64(h*w)/float64(numSuperpixels))), 1)
	nc := float64(40)
	ns := float64(step)

	clusters := make([]int, h*w)
	distances := make([]float64, h*w)
	for i := range clusters {
		clusters[i] = -1
		distances[i] = math.MaxFloat64
	}

	type Center struct{ l, a, b, cx, cy float64 }
	var centers []Center
	for cy := step; cy < h-step/2; cy += step {
		for cx := step; cx < w-step/2; cx += step {
			minGrad := math.MaxFloat64
			lx, ly := cx, cy
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					nx, ny := cx+dx, cy+dy
					if nx < 0 || nx >= w-1 || ny < 0 || ny >= h-1 {
						continue
					}
					off1 := pixOffset(w, nx, ny+1)
					off2 := pixOffset(w, nx+1, ny)
					off3 := pixOffset(w, nx, ny)
					i1 := float64(lab.Pix[off1])
					i2 := float64(lab.Pix[off2])
					i3 := float64(lab.Pix[off3])
					grad := math.Abs(i1-i3) + math.Abs(i2-i3)
					if grad < minGrad {
						minGrad = grad
						lx, ly = nx, ny
					}
				}
			}
			off := pixOffset(w, lx, ly)
			centers = append(centers, Center{
				float64(lab.Pix[off]), float64(lab.Pix[off+1]), float64(lab.Pix[off+2]),
				float64(lx), float64(ly),
			})
		}
	}
	if len(centers) == 0 {
		cx, cy := w/2, h/2
		off := pixOffset(w, cx, cy)
		centers = append(centers, Center{
			float64(lab.Pix[off]), float64(lab.Pix[off+1]), float64(lab.Pix[off+2]),
			float64(cx), float64(cy),
		})
	}

	for range 10 {
		for i := range distances {
			distances[i] = math.MaxFloat64
		}
		for ci, c := range centers {
			x0, x1 := int(c.cx)-step, int(c.cx)+step
			y0, y1 := int(c.cy)-step, int(c.cy)+step
			if x0 < 0 {
				x0 = 0
			}
			if x1 > w {
				x1 = w
			}
			if y0 < 0 {
				y0 = 0
			}
			if y1 > h {
				y1 = h
			}
			for y := y0; y < y1; y++ {
				for x := x0; x < x1; x++ {
					labOff := pixOffset(w, x, y)
					dL := float64(lab.Pix[labOff]) - c.l
					dA := float64(lab.Pix[labOff+1]) - c.a
					dB := float64(lab.Pix[labOff+2]) - c.b
					dx := float64(x) - c.cx
					dy := float64(y) - c.cy
					dc := math.Sqrt(dL*dL + dA*dA + dB*dB)
					ds := math.Sqrt(dx*dx + dy*dy)
					d := math.Sqrt((dc/nc)*(dc/nc) + (ds/ns)*(ds/ns))
					pIdx := labelOffset(w, x, y)
					if d < distances[pIdx] {
						distances[pIdx] = d
						clusters[pIdx] = ci
					}
				}
			}
		}
		type Acc struct {
			l, a, b, sx, sy float64
			n               int
		}
		acc := make([]Acc, len(centers))
		for y := range h {
			for x := range w {
				pIdx := labelOffset(w, x, y)
				ci := clusters[pIdx]
				if ci >= 0 {
					labOff := pixOffset(w, x, y)
					acc[ci].l += float64(lab.Pix[labOff])
					acc[ci].a += float64(lab.Pix[labOff+1])
					acc[ci].b += float64(lab.Pix[labOff+2])
					acc[ci].sx += float64(x)
					acc[ci].sy += float64(y)
					acc[ci].n++
				}
			}
		}
		for ci := range centers {
			if acc[ci].n > 0 {
				n := float64(acc[ci].n)
				centers[ci] = Center{acc[ci].l / n, acc[ci].a / n, acc[ci].b / n, acc[ci].sx / n, acc[ci].sy / n}
			}
		}
	}

	// Connectivity enforcement
	lims := max((h*w)/len(centers), 1)
	dx4 := []int{-1, 0, 1, 0}
	dy4 := []int{0, -1, 0, 1}
	newClusters := make([]int, h*w)
	for i := range newClusters {
		newClusters[i] = -1
	}
	label := 0
	for y := range h {
		for x := range w {
			start := labelOffset(w, x, y)
			if newClusters[start] != -1 {
				continue
			}

			elems := make([]int, 1, 64)
			elems[0] = start
			newClusters[start] = label
			adjLabel := label
			for k := range 4 {
				nx, ny := x+dx4[k], y+dy4[k]
				if nx >= 0 && nx < w && ny >= 0 && ny < h {
					nIdx := labelOffset(w, nx, ny)
					if newClusters[nIdx] >= 0 {
						adjLabel = newClusters[nIdx]
						break
					}
				}
			}
			for c := 0; c < len(elems); c++ {
				cur := elems[c]
				cx := cur % w
				cy := cur / w
				for k := range 4 {
					nx, ny := cx+dx4[k], cy+dy4[k]
					if nx >= 0 && nx < w && ny >= 0 && ny < h {
						nIdx := labelOffset(w, nx, ny)
						if newClusters[nIdx] == -1 && clusters[cur] == clusters[nIdx] {
							newClusters[nIdx] = label
							elems = append(elems, nIdx)
						}
					}
				}
			}
			if len(elems) <= lims>>2 {
				for _, e := range elems {
					newClusters[e] = adjLabel
				}
				label--
			}
			label++
		}
	}

	lb.Clusters = labelImage{
		W:      w,
		H:      h,
		Labels: newClusters,
	}
}

// ============ SUPERPIXELS ============

// computeSuperpixels: [L, a, b, cx, cy, R, G, B]  (R,G,B in [0,1])
func (lb *LayerBuilder) computeSuperpixels() {
	labData := lb.Lab
	rgbData := lb.Rgb
	clusters := lb.Clusters
	h := labData.H
	w := labData.W

	maxLabel := -1
	for _, v := range clusters.Labels {
		if v > maxLabel {
			maxLabel = v
		}
	}
	if maxLabel < 0 {
		lb.SuperPixels = nil
		return
	}
	n := maxLabel + 1

	acc := make([]accumulator, n)
	for y := range h {
		for x := range w {
			pIdx := labelOffset(w, x, y)
			label := clusters.Labels[pIdx]
			if label < 0 || label >= n {
				continue
			}
			off := pixOffset(w, x, y)
			acc[label].l += float64(labData.Pix[off])
			acc[label].a += float64(labData.Pix[off+1])
			acc[label].b += float64(labData.Pix[off+2])
			acc[label].cx += float64(x)
			acc[label].cy += float64(y)
			acc[label].r += float64(rgbData.Pix[off]) / 255.0
			acc[label].g += float64(rgbData.Pix[off+1]) / 255.0
			acc[label].bl += float64(rgbData.Pix[off+2]) / 255.0
			acc[label].count++
		}
	}

	spixels := make([]superpixel, n)
	for i := range n {
		if acc[i].count == 0 {
			continue
		}
		c := float64(acc[i].count)
		spixels[i] = superpixel{
			CX:  float32(acc[i].cx / c),
			CY:  float32(acc[i].cy / c),
			lab: colorful.Color{R: acc[i].l / c, G: acc[i].a / c, B: acc[i].b / c},
			rgb: colorful.Color{R: acc[i].r / c, G: acc[i].g / c, B: acc[i].bl / c},
		}
	}
	lb.SuperPixels = spixels
}

// ============ LLE WEIGHT MATRIX (S×S) ============

// Komşular spatial mesafeyle, ağırlıklar renk farkı üzerinden LLE ile hesaplanır.
func (lb *LayerBuilder) buildLLEWeightMatrix(K int) {
	spixels := lb.SuperPixels
	S := len(spixels)
	if S == 0 {
		lb.LLEWeightMatrix = nil
		return
	}
	if K <= 0 {
		K = 1
	}
	if K >= S {
		K = S - 1
	}
	W := mat.NewDense(S, S, nil)
	Wraw := W.RawMatrix()
	Wdata := Wraw.Data
	Wstride := Wraw.Stride

	spX := make([]float64, S)
	spY := make([]float64, S)
	spR := make([]float64, S)
	spG := make([]float64, S)
	spB := make([]float64, S)
	for i := range S {
		spX[i] = float64(spixels[i].CX)
		spY[i] = float64(spixels[i].CY)
		spR[i] = float64(spixels[i].rgb.R)
		spG[i] = float64(spixels[i].rgb.G)
		spB[i] = float64(spixels[i].rgb.B)
	}

	topIdx := make([]int, K)
	topDist := make([]float64, K)
	dr := make([]float64, K)
	dg := make([]float64, K)
	db := make([]float64, K)
	weights := make([]float64, K)
	cBuf := make([]float64, K*K)
	rhsBuf := make([]float64, K)

	for i := range S {
		count := 0
		maxPos, maxDist := 0, -1.0
		xi, yi := spX[i], spY[i]
		for j := range S {
			if i == j {
				continue
			}
			dx := xi - spX[j]
			dy := yi - spY[j]
			d := dx*dx + dy*dy

			if count < K {
				topIdx[count] = j
				topDist[count] = d
				if d > maxDist {
					maxDist = d
					maxPos = count
				}
				count++
				continue
			}
			if d < maxDist {
				topIdx[maxPos] = j
				topDist[maxPos] = d
				maxPos = 0
				maxDist = topDist[0]
				for k := 1; k < K; k++ {
					if topDist[k] > maxDist {
						maxDist = topDist[k]
						maxPos = k
					}
				}
			}
		}

		for j := range K {
			idx := topIdx[j]
			dr[j] = spR[idx] - spR[i]
			dg[j] = spG[idx] - spG[i]
			db[j] = spB[idx] - spB[i]
		}

		ok := solveLLEWeights(dr, dg, db, K, weights, cBuf, rhsBuf)
		row := i * Wstride
		if !ok {
			uniform := 1.0 / float64(K)
			for a := range K {
				Wdata[row+topIdx[a]] = uniform
			}
			continue
		}
		for a := range K {
			Wdata[row+topIdx[a]] = weights[a]
		}
	}
	lb.LLEWeightMatrix = W
}

// ============ ALPHA COMPOSITING SOLVE ============

// Solve superpixel alpha layers with normal alpha compositing (source-over):
// - background channel is opaque and fixed
// - foreground channels are alpha in [0,1] (sigmoid parameterization)
// - objective = reconstruction + manifold smoothness + alpha regularization
func (lb *LayerBuilder) solveFullSystem(lm, lr, lu float64) {
	spixels := lb.SuperPixels
	W := lb.LLEWeightMatrix
	S := len(spixels)
	N := len(lb.Palette)
	if S == 0 || N == 0 {
		lb.L = nil
		return
	}
	M := max(N-1, 0)

	Lmat := mat.NewDense(S, N, nil)
	if M == 0 {
		for s := range S {
			Lmat.Set(s, 0, 1.0)
		}
		lb.L = Lmat
		return
	}

	Wraw := W.RawMatrix()
	Wdata := Wraw.Data
	Wstride := Wraw.Stride

	bgR := float64(lb.Palette[0].R)
	bgG := float64(lb.Palette[0].G)
	bgB := float64(lb.Palette[0].B)

	palR := make([]float64, M)
	palG := make([]float64, M)
	palB := make([]float64, M)
	for k := range M {
		ch := k + 1
		palR[k] = float64(lb.Palette[ch].R)
		palG[k] = float64(lb.Palette[ch].G)
		palB[k] = float64(lb.Palette[ch].B)
	}

	varCount := S * M
	u := make([]float64, varCount)
	alpha := make([]float64, varCount)
	gradA := make([]float64, varCount)
	gradU := make([]float64, varCount)
	m1 := make([]float64, varCount)
	m2 := make([]float64, varCount)

	for s := range S {
		tr := float64(spixels[s].rgb.R)
		tg := float64(spixels[s].rgb.G)
		tb := float64(spixels[s].rgb.B)
		best := 0
		bestD := math.MaxFloat64
		for ch := range N {
			dR := tr - float64(lb.Palette[ch].R)
			dG := tg - float64(lb.Palette[ch].G)
			dB := tb - float64(lb.Palette[ch].B)
			d := dR*dR + dG*dG + dB*dB
			if d < bestD {
				bestD = d
				best = ch
			}
		}
		for k := range M {
			a0 := 0.02
			if (k+1) == best && best != 0 {
				a0 = 0.65
			}
			a0 = min(0.999, max(0.001, a0))
			u[s*M+k] = math.Log(a0 / (1.0 - a0))
		}
	}

	diff := make([]float64, S)
	tmp := make([]float64, S)
	prefR := make([]float64, M+1)
	prefG := make([]float64, M+1)
	prefB := make([]float64, M+1)
	suf := make([]float64, M)

	iters := 140
	lrAdam := 0.05
	beta1 := 0.9
	beta2 := 0.999
	adamEps := 1e-8
	for it := 1; it <= iters; it++ {
		for i := range varCount {
			alpha[i] = 1.0 / (1.0 + math.Exp(-u[i]))
			gradA[i] = 0
		}

		lossRecon := 0.0
		for s := range S {
			row := s * M
			prefR[0], prefG[0], prefB[0] = bgR, bgG, bgB
			for k := range M {
				a := alpha[row+k]
				oneMinus := 1.0 - a
				prefR[k+1] = a*palR[k] + oneMinus*prefR[k]
				prefG[k+1] = a*palG[k] + oneMinus*prefG[k]
				prefB[k+1] = a*palB[k] + oneMinus*prefB[k]
			}
			outR, outG, outB := prefR[M], prefG[M], prefB[M]
			tr := float64(spixels[s].rgb.R)
			tg := float64(spixels[s].rgb.G)
			tb := float64(spixels[s].rgb.B)
			errR := outR - tr
			errG := outG - tg
			errB := outB - tb
			lossRecon += lr * (errR*errR + errG*errG + errB*errB)

			suf[M-1] = 1.0
			for k := M - 2; k >= 0; k-- {
				suf[k] = suf[k+1] * (1.0 - alpha[row+k+1])
			}
			for k := range M {
				dR := (palR[k] - prefR[k]) * suf[k]
				dG := (palG[k] - prefG[k]) * suf[k]
				dB := (palB[k] - prefB[k]) * suf[k]
				gradA[row+k] += 2.0 * lr * (errR*dR + errG*dG + errB*dB)
			}
		}

		lossManifold := 0.0
		for k := range M {
			for s := range S {
				rowW := s * Wstride
				sum := 0.0
				for j := range S {
					sum += Wdata[rowW+j] * alpha[j*M+k]
				}
				diff[s] = alpha[s*M+k] - sum
				lossManifold += lm * diff[s] * diff[s]
			}
			for i := range S {
				sum := 0.0
				for s := range S {
					sum += diff[s] * Wdata[s*Wstride+i]
				}
				tmp[i] = sum
			}
			for s := range S {
				gradA[s*M+k] += 2.0 * lm * (diff[s] - tmp[s])
			}
		}

		lossReg := 0.0
		for i := range varCount {
			a := alpha[i]
			lossReg += lu * a * a
			gradA[i] += 2.0 * lu * a
		}

		for i := range varCount {
			a := alpha[i]
			gradU[i] = gradA[i] * a * (1.0 - a)
		}

		b1t := 1.0 - math.Pow(beta1, float64(it))
		b2t := 1.0 - math.Pow(beta2, float64(it))
		for i := range varCount {
			g := gradU[i]
			m1[i] = beta1*m1[i] + (1.0-beta1)*g
			m2[i] = beta2*m2[i] + (1.0-beta2)*g*g
			mhat := m1[i] / b1t
			vhat := m2[i] / b2t
			u[i] -= lrAdam * mhat / (math.Sqrt(vhat) + adamEps)
		}

		if it == 1 || it == iters || it%35 == 0 {
			fmt.Printf("   over-solve iter %d/%d loss=%.6f (recon=%.6f manifold=%.6f reg=%.6f)\n",
				it, iters, lossRecon+lossManifold+lossReg, lossRecon, lossManifold, lossReg)
		}
	}

	for s := range S {
		Lmat.Set(s, 0, 1.0)
		row := s * M
		for k := range M {
			ch := k + 1
			a := 1.0 / (1.0 + math.Exp(-u[row+k]))
			Lmat.Set(s, ch, a)
		}
	}
	lb.L = Lmat
}

// ============ PER-PIXEL LLE + OUTPUT ============

func (lb *LayerBuilder) computePerPixelLLEWeights(Kp int) {
	L := lb.L
	rgbData := lb.Rgb
	spixels := lb.SuperPixels
	h := rgbData.H
	w := rgbData.W
	S := len(spixels)
	if L == nil || S == 0 || h == 0 || w == 0 {
		lb.PixelWeights = nil
		return
	}
	if Kp <= 0 {
		Kp = 1
	}
	if Kp > S {
		Kp = S
	}

	Lraw := L.RawMatrix()
	numChannels := Lraw.Cols
	Ldata := Lraw.Data
	Lstride := Lraw.Stride
	layerCount := max(numChannels-1, 0)

	lb.PixelWeights = make([]float64, h*w*numChannels)

	spX := make([]float64, S)
	spY := make([]float64, S)
	spR := make([]float64, S)
	spG := make([]float64, S)
	spB := make([]float64, S)
	for i := range S {
		spX[i] = float64(spixels[i].CX)
		spY[i] = float64(spixels[i].CY)
		spR[i] = float64(spixels[i].rgb.R)
		spG[i] = float64(spixels[i].rgb.G)
		spB[i] = float64(spixels[i].rgb.B)
	}

	// Uniform spatial grid for fast nearest-superpixel candidate lookup.
	cellSize := math.Sqrt(float64(w*h) / float64(S))
	if cellSize < 1.0 {
		cellSize = 1.0
	}
	gridW := int(math.Ceil(float64(w) / cellSize))
	gridH := int(math.Ceil(float64(h) / cellSize))
	if gridW < 1 {
		gridW = 1
	}
	if gridH < 1 {
		gridH = 1
	}
	gridHead := make([]int, gridW*gridH)
	for i := range gridHead {
		gridHead[i] = -1
	}
	gridNext := make([]int, S)
	for i := range S {
		gx := clampInt(int(spX[i]/cellSize), 0, gridW-1)
		gy := clampInt(int(spY[i]/cellSize), 0, gridH-1)
		cell := gy*gridW + gx
		gridNext[i] = gridHead[cell]
		gridHead[cell] = i
	}

	topIdx := make([]int, Kp)
	topDist := make([]float64, Kp)
	dr := make([]float64, Kp)
	dg := make([]float64, Kp)
	db := make([]float64, Kp)
	weights := make([]float64, Kp)
	cBuf := make([]float64, Kp*Kp)
	rhsBuf := make([]float64, Kp)
	alpha := make([]float64, layerCount)
	alpha0 := make([]float64, layerCount)
	prefR := make([]float64, layerCount+1)
	prefG := make([]float64, layerCount+1)
	prefB := make([]float64, layerCount+1)
	suf := make([]float64, layerCount)
	grad := make([]float64, layerCount)
	palR := make([]float64, layerCount)
	palG := make([]float64, layerCount)
	palB := make([]float64, layerCount)
	for k := range layerCount {
		ch := k + 1
		palR[k] = float64(lb.Palette[ch].R)
		palG[k] = float64(lb.Palette[ch].G)
		palB[k] = float64(lb.Palette[ch].B)
	}
	bgR := float64(lb.Palette[0].R)
	bgG := float64(lb.Palette[0].G)
	bgB := float64(lb.Palette[0].B)

	for y := range h {
		for x := range w {
			pixOff := pixOffset(w, x, y)
			pR := float64(rgbData.Pix[pixOff]) / 255.0
			pG := float64(rgbData.Pix[pixOff+1]) / 255.0
			pB := float64(rgbData.Pix[pixOff+2]) / 255.0

			// Kp nearest superpixels in spatial domain (grid-accelerated).
			px := float64(x)
			py := float64(y)
			centerGX := clampInt(int(px/cellSize), 0, gridW-1)
			centerGY := clampInt(int(py/cellSize), 0, gridH-1)

			count := 0
			maxPos, maxDist := 0, -1.0
			for ring := 0; ; ring++ {
				minGX := max(centerGX-ring, 0)
				maxGX := min(centerGX+ring, gridW-1)
				minGY := max(centerGY-ring, 0)
				maxGY := min(centerGY+ring, gridH-1)

				for gy := minGY; gy <= maxGY; gy++ {
					row := gy * gridW
					for gx := minGX; gx <= maxGX; gx++ {
						if ring > 0 && gx > minGX && gx < maxGX && gy > minGY && gy < maxGY {
							continue
						}

						for i := gridHead[row+gx]; i != -1; i = gridNext[i] {
							dx := px - spX[i]
							dy := py - spY[i]
							d := dx*dx + dy*dy

							if count < Kp {
								topIdx[count] = i
								topDist[count] = d
								if d > maxDist {
									maxDist = d
									maxPos = count
								}
								count++
								continue
							}
							if d < maxDist {
								topIdx[maxPos] = i
								topDist[maxPos] = d
								maxPos = 0
								maxDist = topDist[0]
								for k := 1; k < Kp; k++ {
									if topDist[k] > maxDist {
										maxDist = topDist[k]
										maxPos = k
									}
								}
							}
						}
					}
				}

				if count >= Kp {
					minOutside := minDistanceToOutsideCellsSquared(px, py, minGX, maxGX, minGY, maxGY, gridW, gridH, cellSize)
					if minOutside > maxDist {
						break
					}
				}

				if minGX == 0 && maxGX == gridW-1 && minGY == 0 && maxGY == gridH-1 {
					break
				}
			}

			if count == 0 {
				continue
			}
			if count < Kp {
				last := topIdx[count-1]
				for j := count; j < Kp; j++ {
					topIdx[j] = last
				}
			}

			for j := range Kp {
				idx := topIdx[j]
				dr[j] = spR[idx] - pR
				dg[j] = spG[idx] - pG
				db[j] = spB[idx] - pB
			}
			ok := solveLLEWeights(dr, dg, db, Kp, weights, cBuf, rhsBuf)
			if !ok {
				uniform := 1.0 / float64(Kp)
				for j := range Kp {
					weights[j] = uniform
				}
			}

			pixWOff := (y*w + x) * numChannels
			for l := range numChannels {
				lb.PixelWeights[pixWOff+l] = 0
			}
			for j := range Kp {
				qj := weights[j]
				rowOffset := topIdx[j] * Lstride
				for l := range numChannels {
					lb.PixelWeights[pixWOff+l] += qj * Ldata[rowOffset+l]
				}
			}

			for k := range layerCount {
				ch := k + 1
				a := lb.PixelWeights[pixWOff+ch]
				a = min(1.0, max(0.0, a))
				alpha[k] = a
				alpha0[k] = a
			}
			lb.PixelWeights[pixWOff] = 1.0

			if layerCount > 0 {
				const refineIters = 2
				const step = 0.2
				const prox = 0.15
				for range refineIters {
					prefR[0], prefG[0], prefB[0] = bgR, bgG, bgB
					for k := range layerCount {
						a := alpha[k]
						om := 1.0 - a
						prefR[k+1] = a*palR[k] + om*prefR[k]
						prefG[k+1] = a*palG[k] + om*prefG[k]
						prefB[k+1] = a*palB[k] + om*prefB[k]
					}
					outR, outG, outB := prefR[layerCount], prefG[layerCount], prefB[layerCount]
					errR := outR - pR
					errG := outG - pG
					errB := outB - pB

					suf[layerCount-1] = 1.0
					for k := layerCount - 2; k >= 0; k-- {
						suf[k] = suf[k+1] * (1.0 - alpha[k+1])
					}
					for k := range layerCount {
						dR := (palR[k] - prefR[k]) * suf[k]
						dG := (palG[k] - prefG[k]) * suf[k]
						dB := (palB[k] - prefB[k]) * suf[k]
						grad[k] = 2.0*(errR*dR+errG*dG+errB*dB) + 2.0*prox*(alpha[k]-alpha0[k])
					}
					for k := range layerCount {
						alpha[k] = min(1.0, max(0.0, alpha[k]-step*grad[k]))
					}
				}
				for k := range layerCount {
					ch := k + 1
					lb.PixelWeights[pixWOff+ch] = alpha[k]
				}
			}
		}
	}
}

func solveLLEWeights(dr, dg, db []float64, k int, out, cBuf, rhsBuf []float64) bool {
	trace := 0.0
	for i := range k {
		trace += dr[i]*dr[i] + dg[i]*dg[i] + db[i]*db[i]
	}

	if k > 3 {
		tol := 0.001 * trace
		if tol > 1e-12 {
			invTol := 1.0 / tol

			s00, s01, s02 := 0.0, 0.0, 0.0
			s11, s12, s22 := 0.0, 0.0, 0.0
			v0, v1, v2 := 0.0, 0.0, 0.0
			for i := range k {
				r, g, b := dr[i], dg[i], db[i]
				s00 += r * r
				s01 += r * g
				s02 += r * b
				s11 += g * g
				s12 += g * b
				s22 += b * b
				v0 += r
				v1 += g
				v2 += b
			}

			M := [9]float64{
				1.0 + invTol*s00, invTol * s01, invTol * s02,
				invTol * s01, 1.0 + invTol*s11, invTol * s12,
				invTol * s02, invTol * s12, 1.0 + invTol*s22,
			}
			y := [3]float64{v0, v1, v2}
			if solveLinearSystemInPlace(M[:], y[:], 3) {
				scale := invTol * invTol
				for i := range k {
					out[i] = invTol - scale*(dr[i]*y[0]+dg[i]*y[1]+db[i]*y[2])
				}
				if normalizeWeightsInPlace(out, k) {
					return true
				}
			}
		}
	}

	// Generic fallback: solve (Z^T Z + tol*I) w = 1.
	for a := range k {
		ra, ga, ba := dr[a], dg[a], db[a]
		base := a * k
		diag := ra*ra + ga*ga + ba*ba
		cBuf[base+a] = diag
		for b := a + 1; b < k; b++ {
			v := ra*dr[b] + ga*dg[b] + ba*db[b]
			cBuf[base+b] = v
			cBuf[b*k+a] = v
		}
	}
	if k > 3 {
		tol := 0.001 * trace
		for a := range k {
			cBuf[a*k+a] += tol
		}
	}
	for i := range k {
		rhsBuf[i] = 1.0
	}
	if !solveLinearSystemInPlace(cBuf, rhsBuf, k) {
		return false
	}
	copy(out[:k], rhsBuf[:k])
	return normalizeWeightsInPlace(out, k)
}

func normalizeWeightsInPlace(vals []float64, n int) bool {
	sumW := 0.0
	for i := range n {
		sumW += vals[i]
	}
	if math.Abs(sumW) < 1e-12 {
		return false
	}
	inv := 1.0 / sumW
	for i := range n {
		vals[i] *= inv
	}
	return true
}

func projectToSimplexInPlace(vals []float64) {
	sum := 0.0
	for i := range vals {
		if vals[i] < 0 {
			vals[i] = 0
		}
		sum += vals[i]
	}
	if sum <= 1e-12 {
		u := 1.0 / float64(len(vals))
		for i := range vals {
			vals[i] = u
		}
		return
	}
	inv := 1.0 / sum
	for i := range vals {
		vals[i] *= inv
	}
}

func solveLinearSystemInPlace(A []float64, b []float64, n int) bool {
	for col := range n {
		pivotRow := col
		maxAbs := math.Abs(A[col*n+col])
		for r := col + 1; r < n; r++ {
			v := math.Abs(A[r*n+col])
			if v > maxAbs {
				maxAbs = v
				pivotRow = r
			}
		}
		if maxAbs < 1e-12 {
			return false
		}

		if pivotRow != col {
			rowA := col * n
			rowB := pivotRow * n
			for c := col; c < n; c++ {
				A[rowA+c], A[rowB+c] = A[rowB+c], A[rowA+c]
			}
			b[col], b[pivotRow] = b[pivotRow], b[col]
		}

		pivot := A[col*n+col]
		for r := col + 1; r < n; r++ {
			f := A[r*n+col] / pivot
			A[r*n+col] = 0
			rowR := r * n
			rowC := col * n
			for c := col + 1; c < n; c++ {
				A[rowR+c] -= f * A[rowC+c]
			}
			b[r] -= f * b[col]
		}
	}

	for i := n - 1; i >= 0; i-- {
		sum := b[i]
		row := i * n
		for c := i + 1; c < n; c++ {
			sum -= A[row+c] * b[c]
		}
		diag := A[row+i]
		if math.Abs(diag) < 1e-12 {
			return false
		}
		b[i] = sum / diag
	}
	return true
}
