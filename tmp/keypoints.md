# 確率・統計で押さえておきたいポイント

- [(トップページへ戻る)](index)
- [前半（1～8回）のポイント](keytopics1)
- [後半（9～15回）のポイント](keytopics2)

## 以下を押さえておくこと

1. 仮説検定と区間推定の基本
1. 具体的な検定・推定手法（代表的な統計量とその分布）
1. 基本的用語とその意味

$\rightarrow$ スライド（講義内容や宿題），レポート，各回のポイント（上のリンク先）を確認（教科書も参考にしてください）

上記のうち1については以下のポイントを押さえましょう．

## 1. 仮説検定と区間推定の基本

- 仮説検定の流れ（第4回）
  - 帰無仮説・対立仮説の設定
  - **帰無仮説の下で標本の統計量が従う分布（＊）**
    - この分布を使って標本から $p$ 値もしくは棄却域を求める
  - 棄却するか否かは二通りやり方がある
    - $p$ 値を有意水準と比較
    - 観測された統計量が棄却域に入るか否か
  - 棄却できたとき，できなかった時の結論は？
    - 棄却できなかった時の結論に注意
  - $p$ 値とは何か？
    - これを説明できるように
- 区間推定の流れ（第6回）
  - **標本の統計量が従う分布（＊）**
  - 区間推定の導出の考え方を説明できるか？
    - 母数が含まれる確率が信頼係数（95%など）になる区間を，統計量の観測値を用いて表したい
  - 信頼係数や標本サイズと信頼区間（広い？狭い？）の関係

## （＊）標本の統計量はどんな分布に従うか？

標本抽出を「仮に」何度も行うと，抽出のたびに一つの標本統計量が求まる．この標本統計量はどのような分布に従うだろうか？（$\rightarrow$ 標本平均については第5回，標本比率については第10回予定）
  
- 標本抽出を仮想的に何千回も試して，標本平均（標本統計量のひとつ）の分布をみてみよう
  - $\rightarrow$ 第４，５回，およびレポート2で行った演習がこれ
- 多くの仮説検定や推定手法では，上記のようなシミュレーションではなく，母集団に仮定をおいて（確率論から）導出した標本統計量の分布を用いる
  - $\rightarrow$ 母集団に何を仮定するか，どの統計量を扱うかによって様々な検定・推定がある
  - （例）母集団が正規分布であることを仮定すると，標本平均は正規分布に従う $\rightarrow$ $Z$検定ができる
  - （例）標本サイズ$n$が十分大きいことを仮定すると，標本平均は正規分布に近似的に従う $\rightarrow$ $Z$検定ができる

### 標本平均や標本比率が正規分布になるケースについては導出方法を知っておこう

- 標本平均や標本比率と正規分布
  - 分散既知の母平均の検定（第4回）
  - 母比率の差の検定（第10回予定）

### 代表的なものは統計量と分布の基本的特徴を知っておこう
  
- 標本平均（統計量）と $t$ 分布（分散未知の母平均の検定）（第6回）
- 不偏分散の比（統計量）と $F$ 分布（母分散の検定，分散分析）（第7回予定）
- カイ二乗（統計量）と$\chi^2$ 分布（カイ二乗検定）（第11回予定）
  - （実は $t$ 分布や $F$ 分布も$\chi^2$ 分布から導出される．）

## 練習問題

### 前半

- [昨年の中間テスト](exercise/exam1-2022.pdf)
  - [解答](exercise/exam1-2022_answer.pdf)
- [今年の中間テスト](exercise/exam1-2023.pdf)
  - [解答](exercise/exam1-2023_answer.pdf)

### 後半

- [第9,10回の練習問題](exercise/ex_lec9-10_2023.pdf)
  - [解答](exercise/ex_lec9-10_2023_answer.pdf)

- [第9～14回の練習問題（上の練習問題含む）](exercise/ex_lec9-14_2023.pdf)
  - [解答](exercise/ex_lec9-14_2023_answer.pdf)
  - [解答（スライドページ参照付き）](exercise/ex_lec9-14_2023_answer-c.pdf)

日本語が化ける場合はいったんダウンロードしてAcrobat Readerなどで開いてください．