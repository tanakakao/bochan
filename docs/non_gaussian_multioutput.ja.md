# non-Gaussian 多出力回帰

Bochan は、相関あり multitask と独立 objective の model list を明確に区別します。
Gamma、Poisson、Negative Binomial、Beta は共通の契約に従います。いずれも
non-Gaussian sparse variational model であり、BoTorch の exact Gaussian
`MultiTaskGP` / `KroneckerMultiTaskGP` そのものではありません。

| model type | input format | missing targets | output correlation | intended use |
| --- | --- | ---: | ---: | --- |
| multitask | long + task feature | no | yes | irregular task observations |
| wide multitask | wide `[n, m]` | yes | yes | partially observed wide data |
| Kronecker | wide complete block | no | yes | complete aligned observations |
| model list | wide or per-output | per submodel | no | independent objectives |

`*_multitask` は task id を含む long input と scalar target を要求します。
`*_wide_multitask` は finite な観測セルだけを long 形式へ変換し、欠損値を補完しません。
`*_kronecker` は ICM による分離共分散 `K_x ⊗ K_task` と sparse variational
inference を用いる近似モデルです。`NonGaussianModelList` は各 submodel 固有の
response posterior を `PosteriorList` のまま保持し、出力間相関を導入しません。

support は Beta が `0 < y < 1`、Gamma が `y > 0`、Poisson と Negative
Binomial が非負整数です。wide multitask の場合に限り NaN を除いた観測値を検証します。

registry key は `beta|gamma|poisson|negative_binomial` と `_multitask`、
`_wide_multitask`、`_kronecker` の組合せです。従来の `*_multitask` の wide
入力という意味は breaking change であり、wide data は明示的に
`*_wide_multitask` を指定してください。

fantasize を必要とする獲得関数は、全 submodel が正しく対応するときだけ利用できます。
未対応モデルを Gaussian proxy に置換する fallback は行いません。
