

from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import io

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
# CARGAR DATASET LOCAL DESDE DISCO
# ══════════════════════════════════════════════════════════════

X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

# ESCALADO
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# MODELO KNN
model = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance'
)

model.fit(X_train_scaled, y_train)

accuracy = round(model.score(X_test_scaled, y_test) * 100, 1)

# ══════════════════════════════════════════════════════════════
# EXTRAER FEATURES DE IMAGEN
# ══════════════════════════════════════════════════════════════

def extract_features(img_bytes):

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img = img.resize((128,128))

    a = np.array(img, dtype=float)

    return [[
        a[:,:,0].mean(),
        a[:,:,1].mean(),
        a[:,:,2].mean(),
        a[:,:,0].std(),
        a[:,:,1].std(),
        a[:,:,2].std(),
        img.width / img.height,
        a.mean()
    ]]

# ══════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════

PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">

<title>VERSIÓN 2 — Dataset Local</title>

<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@400;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">

<style>

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

:root{
    --bg:#13071f;
    --card:#241236;
    --accent:#c084fc;
    --gold:#fbbf24;
    --cat:#f472b6;
    --dog:#34d399;
    --text:#f3e8ff;
    --muted:#a78bca;
    --border:#3d2858;
}

body{
    background:var(--bg);
    color:var(--text);
    font-family:'Fraunces',serif;
    min-height:100vh;

    background-image:
    radial-gradient(circle at top left, rgba(192,132,252,.18), transparent 40%),
    radial-gradient(circle at bottom right, rgba(251,191,36,.12), transparent 40%);
}

header{
    text-align:center;
    padding:2rem;
    border-bottom:1px solid var(--border);
}

.badge{
    display:inline-block;
    background:var(--gold);
    color:#1a0a2e;
    font-size:.75rem;
    font-family:'Space Mono',monospace;
    padding:.35rem .9rem;
    border-radius:999px;
    margin-bottom:1rem;
    font-weight:700;
}

header h1{
    font-size:2.5rem;
}

header h1 span{
    color:var(--accent);
}

header p{
    margin-top:.7rem;
    color:var(--muted);
    font-family:'Space Mono',monospace;
    font-size:.85rem;
}

.tags{
    margin-top:1rem;
}

.tag{
    display:inline-block;
    background:var(--card);
    border:1px solid var(--border);
    padding:.35rem .7rem;
    border-radius:6px;
    margin:.2rem;
    font-size:.72rem;
    color:var(--muted);
    font-family:'Space Mono',monospace;
}

main{
    max-width:760px;
    margin:auto;
    padding:2rem;
}

.cards{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:1rem;
    margin-bottom:2rem;
}

.card{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:14px;
    padding:1.4rem;
    text-align:center;
}

.card .n{
    font-size:1.8rem;
    color:var(--gold);
    font-family:'Space Mono',monospace;
    font-weight:700;
}

.card .t{
    margin-top:.4rem;
    font-size:.75rem;
    color:var(--muted);
    text-transform:uppercase;
}

.upload{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:18px;
    padding:2rem;
}

.upload h2{
    color:var(--accent);
    margin-bottom:1rem;
    font-family:'Space Mono',monospace;
    font-size:1rem;
}

.drop{
    border:2px dashed var(--border);
    border-radius:14px;
    padding:3rem 1rem;
    text-align:center;
    cursor:pointer;
    transition:.25s;
}

.drop:hover{
    border-color:var(--accent);
    background:rgba(192,132,252,.08);
}

.drop .icon{
    font-size:4rem;
    margin-bottom:1rem;
}

.drop strong{
    display:block;
    font-size:1.1rem;
    margin-bottom:.5rem;
}

.drop p{
    color:var(--muted);
    font-size:.85rem;
}

.drop input{
    display:none;
}

#previewWrap{
    display:none;
    margin-top:1.5rem;
    text-align:center;
}

#preview{
    max-width:100%;
    max-height:320px;
    border-radius:12px;
    border:2px solid var(--border);
}

button{
    width:100%;
    margin-top:1.5rem;
    padding:1rem;
    border:none;
    border-radius:12px;
    cursor:pointer;
    font-size:1rem;
    font-weight:700;
    background:linear-gradient(135deg,var(--accent),var(--gold));
    color:#1a0a2e;
}

#result{
    display:none;
    margin-top:1.5rem;
    padding:2rem;
    border-radius:16px;
    text-align:center;
}

#result.cat{
    border:2px solid var(--cat);
    background:rgba(244,114,182,.12);
}

#result.dog{
    border:2px solid var(--dog);
    background:rgba(52,211,153,.12);
}

.big{
    font-size:5rem;
}

.verdict{
    font-size:2rem;
    margin-top:.5rem;
    font-weight:700;
}

.sub{
    margin-top:.5rem;
    color:var(--muted);
    font-family:'Space Mono',monospace;
    font-size:.8rem;
}

.bar{
    height:10px;
    background:#3a2754;
    border-radius:999px;
    margin-top:1rem;
    overflow:hidden;
}

.fill{
    height:100%;
    width:0%;
    transition:.5s;
}

.cat .fill{
    background:var(--cat);
}

.dog .fill{
    background:var(--dog);
}

@media(max-width:700px){

.cards{
    grid-template-columns:1fr;
}

header h1{
    font-size:2rem;
}

}

</style>
</head>

<body>

<header>

<div class="badge">VERSIÓN 2</div>

<h1>
Dataset <span>Local</span>
</h1>

<p>
Clasificador KNN · Dataset almacenado en disco duro (.NPY)
</p>

<div class="tags">
<span class="tag">Flask</span>
<span class="tag">KNN</span>
<span class="tag">NumPy</span>
<span class="tag">Dataset Local</span>
<span class="tag">Precisión: {{acc}}%</span>
</div>

</header>

<main>

<div class="cards">

<div class="card">
<div class="n">LOCAL</div>
<div class="t">Tipo Dataset</div>
</div>

<div class="card">
<div class="n">KNN</div>
<div class="t">Algoritmo</div>
</div>

<div class="card">
<div class="n">{{acc}}%</div>
<div class="t">Precisión</div>
</div>

</div>

<div class="upload">

<h2>// SUBIR IMAGEN</h2>

<div class="drop" onclick="document.getElementById('file').click()">

<div class="icon">🐾</div>

<strong>Haz clic para seleccionar</strong>

<p>JPG · PNG · WEBP</p>

<input type="file" id="file" accept="image/*" onchange="loadImage(this.files[0])">

</div>

<div id="previewWrap">
<img id="preview">
</div>

<button onclick="predict()" id="btn">
🔍 Detectar Animal
</button>

<div id="result">

<div class="big" id="emoji"></div>

<div class="verdict" id="label"></div>

<div class="sub" id="sub"></div>

<div class="bar">
<div class="fill" id="fill"></div>
</div>

</div>

</div>

</main>

<script>

let file = null;

function loadImage(f){

    if(!f) return;

    file = f;

    const reader = new FileReader();

    reader.onload = e => {

        document.getElementById('preview').src = e.target.result;

        document.getElementById('previewWrap').style.display = 'block';

        document.getElementById('result').style.display = 'none';
    };

    reader.readAsDataURL(f);
}

async function predict(){

    if(!file) return;

    document.getElementById('btn').innerText = '⏳ Analizando...';

    const fd = new FormData();

    fd.append('image', file);

    const res = await fetch('/predict', {
        method:'POST',
        body:fd
    });

    const data = await res.json();

    const result = document.getElementById('result');

    const isDog = data.label == 1;

    result.className = isDog ? 'dog' : 'cat';

    document.getElementById('emoji').innerText =
        isDog ? '🐶' : '🐱';

    document.getElementById('label').innerText =
        isDog ? '¡Es un PERRO!' : '¡Es un GATO!';

    document.getElementById('sub').innerText =
        'Confianza: ' + data.conf + '%';

    result.style.display = 'block';

    setTimeout(() => {
        document.getElementById('fill').style.width = data.conf + '%';
    },100);

    document.getElementById('btn').innerText =
        '🔍 Detectar Animal';
}

</script>

</body>
</html>
"""

# ══════════════════════════════════════════════════════════════
# RUTA PRINCIPAL
# ══════════════════════════════════════════════════════════════

@app.route("/")
def home():

    return render_template_string(
        PAGE,
        acc=accuracy
    )

# ══════════════════════════════════════════════════════════════
# PREDICCIÓN
# ══════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():

    img_bytes = request.files["image"].read()

    feats = extract_features(img_bytes)

    X = scaler.transform(feats)

    pred = int(model.predict(X)[0])

    prob = model.predict_proba(X)[0]

    confidence = round(float(prob[pred]) * 100, 1)

    return jsonify({
        "label": pred,
        "conf": confidence
    })

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("═══════════════════════════════════════")
    print("VERSIÓN 2 — DATASET LOCAL")
    print("Dataset cargado desde archivos .NPY")
    print(f"Precisión: {accuracy}%")
    print("http://localhost:5002")
    print("═══════════════════════════════════════")

    app.run(debug=True, port=5002)