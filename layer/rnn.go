package layer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

//RNN layer
type RNN struct {
	Wx    *mat.Dense   // weight of input data
	Wh    *mat.Dense   // weight of rnn output
	B     *mat.Dense   // bias
	Grads []*mat.Dense // cache grads
	Cache []*mat.Dense // cache
}

func newRNN(Wx *mat.Dense, Wh *mat.Dense, b *mat.Dense) RNN {
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

func (rnn *RNN) forward(x *mat.Dense, hPrev *mat.Dense) mat.Dense {
	wx, wh, b := rnn.Wx, rnn.Wh, rnn.B
	wD, _ := x.Dims()
	_, hD := wx.Dims()
	// 内積計算
	var xArg, hArg *mat.Dense
	xArg.Mul(wx, x)
	hArg.Mul(wh, hPrev)
	// hidden stateの計算
	t := mat.NewDense(wD, hD, nil)
	t.Add(xArg, hArg)
	t.Add(t, b)
	t.Apply(matTanh, t)
	rnn.Cache = []*mat.Dense{x, hPrev, t}
	return *t
}

func (rnn *RNN) backward(dhNext *mat.Dense) {
	wx, wh, b := rnn.Wx, rnn.Wh, rnn.B
	x, hPrev, hNext := rnn.Cache[0], rnn.Cache[1], rnn.Cache[2]
	var dth, db *mat.Dense
	dth = dtanh(dhNext, hNext)
	db = dbias(dth)
}

// tanh層の逆伝搬計算
func dtanh(dt, y *mat.Dense) *mat.Dense {
	var one, dtr, result *mat.Dense
	one.Apply(fullOne, y)
	dtr.Pow(y, 2)
	dtr.Sub(one, dtr)
	result.Mul(dt, dtr)
	return result
}

// biasの加算ノード逆伝搬
func dbias(dth mat.Matrix) *mat.Dense {
	dst := mat.Col(nil, 0, dth)
	dstc := len(dst)
	return mat.NewDense(1, dstc, dst)
}

// 行列を全て1にする
func fullOne(i, j int, v float64) float64 {
	return 1
}

// 行列に対してtanhを計算する
func matTanh(i, j int, v float64) float64 {
	return math.Tanh(v)
}
