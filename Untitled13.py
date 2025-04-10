#!/usr/bin/env python
# coding: utf-8

# In[2]:


# train_model.py
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import joblib

# データ読み込み
df = pd.read_csv("/Users/nishiodaigo/Downloads/卒業実験2.csv")

# ラベルエンコード
le = LabelEncoder()
df["感情1"] = le.fit_transform(df["感情1"])
df["感情2"] = le.fit_transform(df["感情2"])
df["段階"] = le.fit_transform(df["段階"])

# 特徴量と目的変数
x = df[["年齢", "感情1", "感情2"]]
t = df["段階"]

# モデル学習
model = tree.DecisionTreeClassifier(random_state=0)
model.fit(x, t)

# モデル保存
joblib.dump(model, "model.pkl")
print("モデルを model.pkl に保存しました")

import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ====== 定義エリア ======
感情変換 = {"イライラ": 0, "不快感": 1, "冷静": 2, "喪失感": 3, "後悔": 4, "怒り": 5, "悲しみ": 6, "無し": 7, "罪悪感": 8}
段階逆変換 = {0: "受容", 1: "否認", 2: "怒り", 3: "抑鬱"}

抗がん剤副作用辞書 = {
    "パクリタキセル": "末梢神経障害、脱毛、吐き気、骨髄抑制",
    "カルボプラチン": "骨髄抑制、吐き気、腎機能障害",
    "シスプラチン": "腎障害、聴力障害、吐き気、骨髄抑制",
    "ドセタキセル": "浮腫、末梢神経障害、骨髄抑制",
    "アバスチン": "高血圧、出血、蛋白尿",
}

アドバイス辞書 = {
    ("内向的", "低", "受容"): "この患者さんは一人で気持ちを整理する傾向があり、落ち着いて受け入れようとしています。無理な励ましよりも、そっと寄り添う声かけが有効です。",
    ("内向的", "低", "否認"): "感情表現が苦手で、自分の状態に気づきにくいかもしれません。本人のペースで話せる環境を整えてください。",
    ("内向的", "低", "怒り"): "怒りを外に出しにくいため、内に抱え込みやすい傾向があります。否定せず、書くことや静かな対話を促すと良いです。",
    ("内向的", "低", "抑鬱"): "静かに沈んでいる印象があり、声かけに反応が薄いこともあります。焦らず傍にいる姿勢が支えになります。",
    ("内向的", "高", "受容"): "受け入れてはいますが不安や緊張を内に抱えている可能性があります。安心できる雰囲気作りを意識しましょう。",
    ("内向的", "高", "否認"): "神経質になりがちで、わずかな変化にも敏感です。丁寧で具体的な説明が信頼につながります。",
    ("内向的", "高", "怒り"): "怒りと不安が混ざり、表現が抑えられている場合があります。安心できる空間で気持ちを言語化する支援を。",
    ("内向的", "高", "抑鬱"): "感情の浮き沈みが激しく、孤立感を強く感じやすい傾向があります。定期的な関わりと共感的な姿勢が大切です。",
    ("外向的", "低", "受容"): "前向きで周囲とも良好な関係を築けています。患者中心の情報共有が効果的です。",
    ("外向的", "低", "否認"): "明るく振る舞いながらも現実を受け入れきれていない可能性があります。タイミングを見て事実を整理しましょう。",
    ("外向的", "低", "怒り"): "感情が表に出やすく、看護師や他者に強くあたる場合があります。感情を受け止めつつ距離感を保つことが重要です。",
    ("外向的", "低", "抑鬱"): "普段明るい人ほど抑鬱状態になるとギャップに苦しみます。人前で無理をしないよう伝えると良いでしょう。",
    ("外向的", "高", "受容"): "気丈にふるまいながらも内心は不安定な状態かもしれません。感情を吐き出す機会を意識的に設けましょう。",
    ("外向的", "高", "否認"): "元気そうに見えても実はかなり緊張しています。安心できる人間関係の中で徐々に話を引き出していきましょう。",
    ("外向的", "高", "怒り"): "感情が爆発しやすい傾向があるため、安全な場で感情を受け止めることが大切です。身体的ケアだけでなく精神面にも注目を。",
    ("外向的", "高", "抑鬱"): "不安や落ち込みを他者にぶつけやすくなります。患者の話を途中で遮らず、しっかりと傾聴してください。",
}

try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error("モデルの読み込みに失敗しました。train_model.pyを実行してください。")
    st.stop()

# ====== ページ管理 ======
if "page" not in st.session_state:
    st.session_state.page = "chemo_select"

# ====== 抗がん剤選択画面 ======
if st.session_state.page == "chemo_select":
    st.title("本日使用予定の抗がん剤を選択")
    st.session_state.抗がん剤リスト = st.multiselect("抗がん剤を選択してください：", list(抗がん剤副作用辞書.keys()))

    if st.button("次へ（看護師情報入力へ）"):
        st.session_state.page = "nurse"
        st.rerun()

# ====== 看護師画面 ======
elif st.session_state.page == "nurse":
    st.title("看護師向け診断アプリ")

    st.session_state.名前 = st.text_input("患者さんのお名前")
    st.session_state.年齢 = st.number_input("年齢", min_value=0, step=1)
    st.session_state.生年月日 = st.date_input("生年月日", value=date(1980, 1, 1))
    st.session_state.体温 = st.number_input("体温（℃）", min_value=30.0, max_value=45.0, step=0.1)
    st.session_state.血圧_上 = st.number_input("収縮期血圧（上）", min_value=50)
    st.session_state.血圧_下 = st.number_input("拡張期血圧（下）", min_value=30)
    st.session_state.白血球数 = st.number_input("白血球数 (/μL)", min_value=0)
    st.session_state.ヘモグロビン = st.number_input("ヘモグロビン (g/dL)", min_value=0.0, step=0.1)

    if st.button("次へ（患者入力へ）"):
        st.session_state.page = "patient"
        st.rerun()

# ====== 患者入力画面 ======
elif st.session_state.page == "patient":
    st.title("患者さん入力画面")
    st.session_state.感情1 = st.selectbox("主な感情", list(感情変換.keys()))
    st.session_state.感情2 = st.selectbox("次に強い感情", list(感情変換.keys()))
    st.session_state.性格 = st.radio("性格", ["内向的", "外向的"])
    st.session_state.神経症傾向 = st.radio("心配ごとが頭から離れにくい方ですか？", ["低", "高"])

    if st.button("入力完了"):
        st.session_state.page = "thank_you"
        st.rerun()

# ====== サンクス画面 ======
elif st.session_state.page == "thank_you":
    st.title("ありがとうございました")
    st.write("iPadを看護師に渡してお待ちください。")
    if st.button("次へ（看護師用）※結果がドクターに送信されます"):
        st.session_state.page = "result"
        st.rerun()

# ====== 結果画面 ======
elif st.session_state.page == "result":
    st.title("診断結果レポート")
    st.subheader("● 投与予定の抗がん剤")
    for drug in st.session_state.抗がん剤リスト:
        st.markdown(f"- **{drug}**：{抗がん剤副作用辞書[drug]}")

    異常 = []
    if st.session_state.体温 < 35.0 or st.session_state.体温 > 38.0:
        異常.append("体温異常")
    if st.session_state.血圧_上 < 90 or st.session_state.血圧_上 > 140 or st.session_state.血圧_下 < 50 or st.session_state.血圧_下 > 90:
        異常.append("血圧異常")
    if st.session_state.白血球数 < 3000:
        異常.append("白血球数低値")
    if st.session_state.ヘモグロビン < 10.0:
        異常.append("ヘモグロビン低値")

    感情1_num = 感情変換[st.session_state.感情1]
    感情2_num = 感情変換[st.session_state.感情2]

    try:
        段階_num = model.predict([[st.session_state.年齢, 感情1_num, 感情2_num]])[0]
        段階名 = 段階逆変換.get(段階_num, "不明")
    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {str(e)}")
        st.stop()

    アドバイス = アドバイス辞書.get((st.session_state.性格, st.session_state.神経症傾向, 段階名), "該当するアドバイスが見つかりませんでした。")

    if 異常:
        st.error("本日は抗がん剤投与不可：")
        for abn in 異常:
            st.warning(abn)
    else:
        st.success("抗がん剤投与可能です。")

    st.subheader("● 感情・性格評価")
    st.markdown(f"- 感情段階：**{段階名}**")
    st.markdown(f"- 性格傾向：**{st.session_state.性格}**")
    st.markdown(f"- アドバイス：**{アドバイス}**")

    df = pd.DataFrame([{
        "名前": st.session_state.名前, "年齢": st.session_state.年齢, "生年月日": st.session_state.生年月日,
        "体温": st.session_state.体温, "血圧_上": st.session_state.血圧_上, "血圧_下": st.session_state.血圧_下,
        "白血球数": st.session_state.白血球数, "ヘモグロビン": st.session_state.ヘモグロビン,
        "感情1": st.session_state.感情1, "感情2": st.session_state.感情2, "性格": st.session_state.性格, "神経症傾向": st.session_state.神経症傾向,
        "段階": 段階名, "アドバイス": アドバイス,
        "抗がん剤": ", ".join(st.session_state.抗がん剤リスト),
        "投与可否": "不可" if 異常 else "可"
    }])

    try:
        既存 = pd.read_excel("患者データ.xlsx")
        df = pd.concat([既存, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_excel("患者データ改改.xlsx", index=False)
    st.success("診断結果をExcelに保存しました。")


# In[ ]:




