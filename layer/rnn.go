package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

//RNN layer
type RNN struct {
	Wx    mat.Dense   // weight of input data
	Wh    mat.Dense   // weight of rnn output
	B     mat.Dense   // bias
	Grads []mat.Dense // initial grads from Wx, Wh, b
	cache []mat.Dense
}

func newRNN(Wx mat.Dense, Wh mat.Dense, b mat.Dense) RNN {
	rnn := RNN{Wx: Wx, Wh: Wh, B: b}
	// initialize grads
	Wx.Zero()
	Wh.Zero()
	b.Zero()
	rnn.Grads = append(rnn.Grads, Wx)
	rnn.Grads = append(rnn.Grads, Wh)
	rnn.Grads = append(rnn.Grads, b)
	return rnn
}

func (rnn *RNN) forward(x mat.Dense, hPrev mat.Dense) mat.Dense {
	wx := rnn.Wx
	wh := rnn.Wh
	b := rnn.B
	wD, _ := x.Dims()
	_, hD := wx.Dims()
	// 内積計算
	xArg := mat.NewDense(wD, hD, nil)
	xArg.Product(&wx, &x)
	hArg := mat.NewDense(wD, hD, nil)
	hArg.Product(&wh, &hPrev)
	// hidden stateの計算
	t := mat.NewDense(wD, hD, nil)
	t.Add(xArg, hArg)
	t.Add(t, &b)
	t.Apply(matTanh, t)
	return *t
}

func matTanh(i, j int, v float64) float64 {
	return math.Tanh(v)
}
