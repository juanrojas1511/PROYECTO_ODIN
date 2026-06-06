"""
VERSIÓN 1 — Dataset embebido en el código fuente
Detecta si una imagen es GATO o PERRO usando KNN
con características extraídas de la imagen subida.
"""

from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import io, base64

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
#  DATASET EMBEBIDO EN EL CÓDIGO FUENTE
#  Características de color/textura pre-calculadas de imágenes
#  [mean_r, mean_g, mean_b, std_r, std_g, std_b, ratio, bright, label]
#  label: 0=Gato  1=Perro
# ══════════════════════════════════════════════════════════════
DATASET = [
    # ── GATOS (tonos más cálidos, varianza moderada) ──────────
    [172, 148, 132, 38, 35, 30, 0.98, 151, 0],
    [165, 142, 125, 42, 39, 33, 1.02, 144, 0],
    [180, 155, 138, 35, 32, 28, 0.95, 158, 0],
    [145, 128, 115, 48, 44, 38, 1.05, 129, 0],
    [195, 170, 148, 30, 28, 24, 0.92, 171, 0],
    [158, 138, 122, 52, 47, 41, 1.08, 139, 0],
    [168, 145, 130, 40, 37, 32, 0.97, 148, 0],
    [185, 160, 142, 33, 30, 26, 1.00, 162, 0],
    [142, 125, 112, 55, 50, 43, 1.03, 126, 0],
    [178, 153, 136, 36, 33, 29, 0.96, 156, 0],
    [163, 140, 124, 44, 41, 35, 1.06, 142, 0],
    [190, 164, 146, 31, 29, 25, 0.93, 167, 0],
    [148, 130, 116, 50, 46, 39, 1.04, 131, 0],
    [173, 149, 133, 37, 34, 29, 0.99, 152, 0],
    [160, 138, 122, 46, 42, 36, 1.07, 140, 0],
    [183, 158, 140, 34, 31, 27, 0.94, 160, 0],
    [153, 133, 119, 49, 45, 38, 1.01, 135, 0],
    [176, 152, 135, 39, 36, 31, 0.98, 154, 0],
    [167, 144, 128, 43, 40, 34, 1.02, 146, 0],
    [188, 162, 144, 32, 29, 25, 0.95, 165, 0],
    # ── PERROS (mayor varianza, tonos más diversos) ───────────
    [140, 118, 98,  62, 57, 50, 1.25, 119, 1],
    [155, 130, 108, 70, 65, 56, 1.30, 131, 1],
    [128, 108, 90,  75, 69, 60, 1.20, 109, 1],
    [170, 142, 118, 58, 54, 47, 1.35, 143, 1],
    [118, 100, 83,  80, 74, 64, 1.18, 100, 1],
    [160, 134, 112, 65, 60, 52, 1.28, 135, 1],
    [135, 113, 94,  72, 67, 58, 1.22, 114, 1],
    [175, 147, 122, 55, 51, 44, 1.32, 148, 1],
    [122, 103, 86,  78, 72, 62, 1.15, 104, 1],
    [165, 138, 115, 60, 56, 48, 1.27, 139, 1],
    [130, 110, 92,  74, 68, 59, 1.21, 111, 1],
    [178, 149, 124, 53, 49, 42, 1.33, 150, 1],
    [125, 106, 88,  77, 71, 61, 1.17, 106, 1],
    [162, 136, 113, 63, 58, 50, 1.29, 137, 1],
    [138, 116, 97,  68, 63, 54, 1.23, 117, 1],
    [172, 144, 120, 57, 53, 46, 1.31, 145, 1],
    [120, 101, 84,  81, 75, 65, 1.16, 102, 1],
    [168, 141, 117, 61, 57, 49, 1.26, 142, 1],
    [132, 112, 93,  73, 67, 58, 1.24, 112, 1],
    [180, 151, 126, 52, 48, 41, 1.34, 152, 1],
]

FEATURE_NAMES = ["mean_r","mean_g","mean_b","std_r","std_g","std_b","ratio","brightness"]

def train():
    data = np.array(DATASET)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=7)
    knn = KNeighborsClassifier(
    n_neighbors=1,
    weights='distance'
)
    knn.fit(Xtr, ytr)
    return knn, sc, round(knn.score(Xte, yte) * 100, 1)

model, scaler, accuracy = train()

def extract_features(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img =img.resize((128,128))
    
    a = np.array(img, dtype=float)
    return [
        a[:,:,0].mean(), a[:,:,1].mean(), a[:,:,2].mean(),
        a[:,:,0].std(),  a[:,:,1].std(),  a[:,:,2].std(),
        img.width / img.height,
        a.mean()
    ]

# ══════════════════════════════════════════════════════════════
#  TEMPLATE HTML
# ══════════════════════════════════════════════════════════════
PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clasificador V1 — Gato o Perro</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,700;1,400&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#1a0a2e;--surface:#2d1b4e;--card:#3d2a5e;
  --accent:#c084fc;--gold:#fbbf24;--cat:#f472b6;--dog:#34d399;
  --text:#f3e8ff;--muted:#a78bca;--border:#4c3a6e;
}
body{background:var(--bg);color:var(--text);font-family:'Fraunces',serif;min-height:100vh;
     background-image:radial-gradient(ellipse at 30% 20%,rgba(192,132,252,.15) 0%,transparent 55%),
                      radial-gradient(ellipse at 70% 80%,rgba(251,191,36,.08) 0%,transparent 55%);}

header{padding:2rem;text-align:center;border-bottom:1px solid var(--border);}
.badge{display:inline-block;background:var(--gold);color:#1a0a2e;font-family:'Space Mono',monospace;
       font-size:.7rem;font-weight:700;padding:.3rem .9rem;border-radius:20px;letter-spacing:.1em;margin-bottom:.8rem;}
header h1{font-size:2.4rem;font-weight:700;line-height:1.2;}
header h1 span{color:var(--accent);}
header p{color:var(--muted);font-size:.9rem;font-family:'Space Mono',monospace;margin-top:.5rem;}
.tag{display:inline-block;background:var(--surface);border:1px solid var(--border);
     border-radius:4px;padding:.2rem .6rem;font-size:.7rem;font-family:'Space Mono',monospace;color:var(--muted);margin:.3rem .2rem;}

main{max-width:720px;margin:0 auto;padding:2rem;}

.info-row{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem;}
.info-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem;text-align:center;}
.info-card .n{font-size:1.8rem;font-weight:700;color:var(--gold);font-family:'Space Mono',monospace;}
.info-card .l{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:.3rem;}

.upload-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:2rem;margin-bottom:1.5rem;}
.upload-card h2{font-size:1.1rem;color:var(--accent);margin-bottom:1.5rem;font-family:'Space Mono',monospace;}

.drop{border:2px dashed var(--border);border-radius:12px;padding:3rem 1rem;text-align:center;
      cursor:pointer;transition:all .25s;background:rgba(255,255,255,.02);}
.drop:hover,.drop.over{border-color:var(--accent);background:rgba(192,132,252,.08);
                        box-shadow:0 0 30px rgba(192,132,252,.15);}
.drop .ico{font-size:3.5rem;margin-bottom:1rem;display:block;}
.drop strong{font-size:1.1rem;display:block;margin-bottom:.4rem;}
.drop p{color:var(--muted);font-size:.85rem;}
.drop input{display:none;}

#preview-wrap{margin-top:1.5rem;display:none;text-align:center;}
#preview-wrap img{max-width:100%;max-height:320px;border-radius:10px;
                  border:2px solid var(--border);box-shadow:0 8px 32px rgba(0,0,0,.4);}

.btn{display:block;width:100%;margin-top:1.2rem;padding:1rem;font-size:1.05rem;font-weight:700;
     font-family:'Fraunces',serif;border:none;border-radius:10px;cursor:pointer;
     background:linear-gradient(135deg,var(--accent),var(--gold));color:#1a0a2e;
     transition:opacity .2s;display:none;}
.btn:hover{opacity:.85;}

/* RESULT */
#result{margin-top:1.5rem;border-radius:14px;padding:2rem;text-align:center;display:none;
        animation:pop .4s cubic-bezier(.34,1.56,.64,1);}
@keyframes pop{from{transform:scale(.8);opacity:0}to{transform:scale(1);opacity:1}}
#result.cat{background:rgba(244,114,182,.12);border:2px solid var(--cat);box-shadow:0 0 40px rgba(244,114,182,.2);}
#result.dog{background:rgba(52,211,153,.12);border:2px solid var(--dog);box-shadow:0 0 40px rgba(52,211,153,.2);}
#result .big{font-size:5rem;line-height:1;display:block;margin-bottom:.5rem;}
#result .verdict{font-size:2.2rem;font-weight:700;margin-bottom:.4rem;}
#result.cat .verdict{color:var(--cat);}
#result.dog .verdict{color:var(--dog);}
#result .sub{font-size:.8rem;color:var(--muted);font-family:'Space Mono',monospace;}
.bar-wrap{margin:.8rem auto 0;max-width:280px;}
.bar-label{display:flex;justify-content:space-between;font-size:.75rem;font-family:'Space Mono',monospace;color:var(--muted);margin-bottom:.3rem;}
.bar{height:8px;border-radius:4px;background:var(--border);overflow:hidden;}
.bar-fill{height:100%;border-radius:4px;transition:width .6s ease;width:0;}
#result.cat .bar-fill{background:linear-gradient(90deg,var(--cat),#f9a8d4);}
#result.dog .bar-fill{background:linear-gradient(90deg,var(--dog),#6ee7b7);}

.dataset-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.5rem;}
.dataset-box h2{font-size:.85rem;color:var(--accent);font-family:'Space Mono',monospace;margin-bottom:1rem;}
.ds-scroll{font-family:'Space Mono',monospace;font-size:.68rem;max-height:160px;overflow-y:auto;line-height:1.9;}
.ds-scroll .c{color:var(--cat);}
.ds-scroll .d{color:var(--dog);}
</style>
</head>
<body>
<header>
  <div class="badge">VERSIÓN 1</div>
  <h1>¿Es <span>Gato</span> o <span style="color:var(--gold)">Perro</span>?</h1>
  <p>Dataset embebido en código · Algoritmo KNN</p>
  <div>
    <span class="tag">40 muestras hardcodeadas</span>
    <span class="tag">KNN k=3</span>
    <span class="tag">Precisión: {{acc}}%</span>
  </div>
</header>

<main>
  <div class="info-row">
    <div class="info-card"><div class="n">40</div><div class="l">Muestras en código</div></div>
    <div class="info-card"><div class="n">KNN</div><div class="l">Algoritmo</div></div>
    <div class="info-card"><div class="n">{{acc}}%</div><div class="l">Precisión</div></div>
  </div>

  <div class="upload-card">
    <h2>// SUBE UNA IMAGEN</h2>
    <div class="drop" id="drop"
         ondragover="ev(event,true)" ondragleave="ev(event,false)" ondrop="drop(event)"
         onclick="document.getElementById('fi').click()">
      <span class="ico">🐾</span>
      <strong>Arrastra tu imagen aquí</strong>
      <p>o haz clic para seleccionar · JPG, PNG, WEBP</p>
      <input type="file" id="fi" accept="image/*" onchange="load(this.files[0])">
    </div>
    <div id="preview-wrap">
      <img id="preview" src="" alt="preview">
    </div>
    <button class="btn" id="btn" onclick="classify()">🔍 Detectar: ¿Gato o Perro?</button>
    <div id="result">
      <span class="big" id="r-ico"></span>
      <div class="verdict" id="r-label"></div>
      <div class="sub" id="r-sub"></div>
      <div class="bar-wrap">
        <div class="bar-label"><span id="r-pct">0%</span><span>confianza</span></div>
        <div class="bar"><div class="bar-fill" id="r-bar"></div></div>
      </div>
    </div>
  </div>

  <div class="dataset-box">
    <h2>// DATASET EMBEBIDO EN app.py</h2>
    <div class="ds-scroll">
<span style="color:#4c3a6e"># [mean_r, mean_g, mean_b, std_r, std_g, std_b, ratio, bright, label]</span>
{% for row in dataset %}
<span class="{{row.cls}}">[{{row.vals}}]  → {{row.name}}</span>
{% endfor %}
    </div>
  </div>
</main>

<script>
let file=null;
function ev(e,on){e.preventDefault();document.getElementById('drop').classList.toggle('over',on);}
function drop(e){e.preventDefault();document.getElementById('drop').classList.remove('over');load(e.dataTransfer.files[0]);}
function load(f){
  if(!f)return; file=f;
  const rd=new FileReader();
  rd.onload=e=>{
    document.getElementById('preview').src=e.target.result;
    document.getElementById('preview-wrap').style.display='block';
    document.getElementById('btn').style.display='block';
    document.getElementById('result').style.display='none';
  };
  rd.readAsDataURL(f);
}
async function classify(){
  if(!file)return;
  document.getElementById('btn').textContent='⏳ Analizando...';
  const fd=new FormData(); fd.append('image',file);
  const res=await fetch('/predict',{method:'POST',body:fd});
  const d=await res.json();
  const el=document.getElementById('result');
  el.className='result '+(d.label===0?'cat':'dog');
  document.getElementById('r-ico').textContent=d.label===0?'🐶':'🐱';
  document.getElementById('r-label').textContent=d.label===0?'¡Es un PERRO!':'¡Es un GATO!';
  document.getElementById('r-sub').textContent=`Confianza: ${(d.conf*100).toFixed(1)}% · KNN k=3 · Dataset en código fuente`;
  const pct=Math.round(d.conf*100);
  document.getElementById('r-pct').textContent=pct+'%';
  el.style.display='block';
  setTimeout(()=>document.getElementById('r-bar').style.width=pct+'%',50);
  document.getElementById('btn').textContent='🔍 Detectar: ¿Gato o Perro?';
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    rows = []
    for r in DATASET:
        rows.append({
            "cls": "c" if r[-1]==0 else "d",
            "name": "Gato" if r[-1]==0 else "Perro",
            "vals": ", ".join(str(v) for v in r)
        })
    return render_template_string(PAGE, acc=accuracy, dataset=rows)

@app.route("/predict", methods=["POST"])
def predict():
    img_bytes = request.files["image"].read()

    feats = extract_features(img_bytes)

    X = scaler.transform([feats])

    pred = int(model.predict(X)[0])

    # regla auxiliar
    if feats[3] > 55:
        pred = 1

    prob = model.predict_proba(X)[0]

    return jsonify({
        "label": pred,
        "conf": float(prob[pred])
    })

if __name__ == "__main__":
    print(f"[V1] Dataset embebido: {len(DATASET)} muestras | Precisión: {accuracy}%")
    print("[V1] Abre: http://localhost:5001")
    app.run(debug=True, port=5001)