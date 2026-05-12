/**
 * main.js — CancerML Frontend
 * Breast Cancer Wisconsin — Predictor con Regresión Logística y RNA
 */

const API_URL = 'http://localhost:5000';

const FEAT_NAMES = {
  radius:'Radio', texture:'Textura', perimeter:'Perímetro', area:'Área',
  smoothness:'Suavidad', compactness:'Compacidad', concavity:'Concavidad',
  concave_points:'Puntos Cóncavos', symmetry:'Simetría',
  fractal_dimension:'Dim. Fractal'
};
const GRUPO_LABELS = { '1':'Media', '2':'Error Estándar', '3':'Peor Valor' };

const EJEMPLO = {
  radius1:17.99,texture1:10.38,perimeter1:122.8,area1:1001.0,
  smoothness1:0.1184,compactness1:0.2776,concavity1:0.3001,
  concave_points1:0.1471,symmetry1:0.2419,fractal_dimension1:0.07871,
  radius2:1.095,texture2:0.9053,perimeter2:8.589,area2:153.4,
  smoothness2:0.006399,compactness2:0.04904,concavity2:0.05373,
  concave_points2:0.01587,symmetry2:0.03003,fractal_dimension2:0.006193,
  radius3:25.38,texture3:17.33,perimeter3:184.6,area3:2019.0,
  smoothness3:0.1622,compactness3:0.6656,concavity3:0.7119,
  concave_points3:0.2654,symmetry3:0.4601,fractal_dimension3:0.1189
};

let columnas = [];

// ── Init ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await cargarColumnas();
  bindModeloSelectors();
  bindBotones();
  bindDropzone();
});

// ── Cargar columnas ───────────────────────────────
async function cargarColumnas() {
  try {
    const res  = await fetch(`${API_URL}/columnas`);
    const data = await res.json();
    columnas   = data.columnas;
    generarFormulario(columnas);
  } catch {
    document.getElementById('form-features').innerHTML = `
      <div class="col-12">
        <div class="alert-custom">
          <i class="bi bi-exclamation-triangle"></i>
          No se pudo conectar con Flask en <code>http://localhost:5000</code>.
          Ejecuta: <code>python app.py</code>
        </div>
      </div>`;
  }
}

// ── Generar formulario dinámico ───────────────────
function generarFormulario(cols) {
  const cont = document.getElementById('form-features');
  cont.innerHTML = '';
  const grupos = { '1':[], '2':[], '3':[] };
  cols.forEach(col => {
    const m = col.match(/(\d)$/);
    if (m) grupos[m[1]].push(col);
  });
  ['1','2','3'].forEach(g => {
    const titulo = document.createElement('div');
    titulo.className = 'col-12 mt-2';
    titulo.innerHTML = `<div class="feat-group-title">${GRUPO_LABELS[g]}</div>`;
    cont.appendChild(titulo);
    grupos[g].forEach(col => {
      const base  = col.replace(/\d$/, '');
      const label = FEAT_NAMES[base] || base;
      const div   = document.createElement('div');
      div.className = 'col-md-6';
      div.innerHTML = `
        <div class="mb-3">
          <label for="feat-${col}" class="form-label">
            ${label} <span class="feature-code">${col}</span>
          </label>
          <input type="number" step="any" class="form-control feature-input"
                 id="feat-${col}" data-col="${col}"
                 placeholder="${EJEMPLO[col] ?? '0.00'}" />
        </div>`;
      cont.appendChild(div);
    });
  });
}

// ── Selector de modelo (sincroniza 3 grupos de radios) ─
function bindModeloSelectors() {
  document.querySelectorAll(
    '#model-logreg,#model-mlp,#ind-logreg,#ind-mlp,#lot-logreg,#lot-mlp'
  ).forEach(radio => {
    radio.addEventListener('change', () => {
      const esL = radio.id.includes('logreg');
      ['model','ind','lot'].forEach(pref => {
        document.getElementById(`${pref}-logreg`).checked = esL;
        document.getElementById(`${pref}-mlp`).checked    = !esL;
      });
      actualizarBadge(esL);
    });
  });
}

function getModelo() {
  return document.getElementById('ind-logreg').checked ? 'logreg' : 'mlp';
}

function actualizarBadge(esL) {
  const b = document.getElementById('badge-modelo-hero');
  if (!b) return;
  b.className = esL ? 'badge badge-modelo-logreg' : 'badge badge-modelo-mlp';
  b.innerHTML = esL
    ? '<i class="bi bi-graph-up-arrow me-1"></i> Regresión Logística'
    : '<i class="bi bi-diagram-3 me-1"></i> Red Neuronal (MLP)';
}

// ── Bind botones ──────────────────────────────────
function bindBotones() {
  document.getElementById('btn-predecir-individual')
    ?.addEventListener('click', predecirIndividual);
  document.getElementById('btn-predecir-lotes')
    ?.addEventListener('click', predecirLotes);
  document.getElementById('btn-limpiar')
    ?.addEventListener('click', () => {
      document.querySelectorAll('.feature-input').forEach(i => {
        i.value = ''; i.classList.remove('is-invalid');
      });
      document.getElementById('resultado-individual').innerHTML = emptyState('activity','El resultado aparecerá aquí después de predecir');
    });
  document.getElementById('btn-ejemplo')
    ?.addEventListener('click', () => {
      document.querySelectorAll('.feature-input').forEach(inp => {
        const v = EJEMPLO[inp.dataset.col];
        if (v !== undefined) { inp.value = v; inp.classList.remove('is-invalid'); }
      });
    });
  document.getElementById('btn-descargar-ejemplo')
    ?.addEventListener('click', e => { e.preventDefault(); descargarEjemplo(); });
}

// ── Dropzone ──────────────────────────────────────
function bindDropzone() {
  const dz = document.getElementById('dropzone');
  const fi = document.getElementById('archivo-csv');
  if (!dz || !fi) return;
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) infoArchivo(e.dataTransfer.files[0]);
  });
  fi.addEventListener('change', () => { if (fi.files[0]) infoArchivo(fi.files[0]); });
}

function infoArchivo(file) {
  const el = document.getElementById('info-archivo');
  if (!file.name.endsWith('.csv')) {
    el.innerHTML = `<span style="color:var(--danger);font-size:.78rem"><i class="bi bi-x-circle me-1"></i>Solo archivos .csv</span>`;
    return;
  }
  el.innerHTML = `<span style="color:var(--success);font-size:.78rem"><i class="bi bi-check-circle me-1"></i>${file.name} · ${(file.size/1024).toFixed(1)} KB</span>`;
}

// ════════════════════════════════════════════════════════════
// PREDICCIÓN INDIVIDUAL
// ════════════════════════════════════════════════════════════
async function predecirIndividual() {
  const inputs = document.querySelectorAll('.feature-input');
  const datos  = {};
  let valido   = true;
  inputs.forEach(inp => {
    const v = inp.value.trim();
    if (v === '' || isNaN(parseFloat(v))) { inp.classList.add('is-invalid'); valido = false; }
    else { inp.classList.remove('is-invalid'); datos[inp.dataset.col] = parseFloat(v); }
  });
  if (!valido) { showAlerta('alerta-individual','Completa todos los campos con valores numéricos.'); return; }

  const btn = document.getElementById('btn-predecir-individual');
  setBtnLoading(btn, true);
  try {
    const res  = await fetch(`${API_URL}/predecir`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ modelo: getModelo(), datos })
    });
    const data = await res.json();
    if (data.error) { showAlerta('alerta-individual', data.error); return; }
    renderIndividual(data);
  } catch { showAlerta('alerta-individual','Error al conectar con Flask.'); }
  finally  { setBtnLoading(btn, false); }
}

function renderIndividual(data) {
  const esMal  = data.clase === 1;
  const tipo   = esMal ? 'maligno' : 'benigno';
  const icono  = esMal ? 'bi-exclamation-triangle' : 'bi-check-circle';
  const modelo = data.modelo_usado === 'LOGREG' ? 'Regresión Logística' : 'Red Neuronal';

  document.getElementById('resultado-individual').innerHTML = `
    <div class="result-card">
      <div class="result-header-${tipo}">
        <div class="result-icon result-icon-${tipo[0]}">
          <i class="bi ${icono}"></i>
        </div>
        <div>
          <div class="result-label">Diagnóstico · ${modelo}</div>
          <div class="result-diagnosis result-diagnosis-${tipo[0]}">${data.prediccion}</div>
        </div>
      </div>
      <div class="result-body">
        <div class="label-sm mb-3">Probabilidades</div>
        <div class="prog-label">
          <span><i class="bi bi-check-circle me-1" style="color:var(--success)"></i>Benigno</span>
          <span class="prog-val">${data.probabilidad.Benigno}%</span>
        </div>
        <div class="progress"><div class="progress-bar-b" style="width:${data.probabilidad.Benigno}%"></div></div>
        <div class="prog-label">
          <span><i class="bi bi-exclamation-triangle me-1" style="color:var(--danger)"></i>Maligno</span>
          <span class="prog-val">${data.probabilidad.Maligno}%</span>
        </div>
        <div class="progress"><div class="progress-bar-m" style="width:${data.probabilidad.Maligno}%"></div></div>
        <div class="medico-aviso">
          <i class="bi bi-heart-pulse me-1"></i>
          <strong>Aviso:</strong> Predicción académica. No sustituye diagnóstico médico profesional.
        </div>
      </div>
    </div>`;
}

// ════════════════════════════════════════════════════════════
// PREDICCIÓN POR LOTES
// ════════════════════════════════════════════════════════════
async function predecirLotes() {
  const fi = document.getElementById('archivo-csv');
  if (!fi.files.length) { showAlerta('alerta-lotes','Selecciona un archivo CSV.'); return; }

  const modelo = document.getElementById('lot-logreg').checked ? 'logreg' : 'mlp';
  const fd     = new FormData();
  fd.append('archivo', fi.files[0]);
  fd.append('modelo', modelo);

  const btn = document.getElementById('btn-predecir-lotes');
  setBtnLoading(btn, true);
  try {
    const res  = await fetch(`${API_URL}/predecir-lotes`, { method:'POST', body: fd });
    const data = await res.json();
    if (data.error) { showAlerta('alerta-lotes', data.error); return; }
    renderLotes(data);
  } catch { showAlerta('alerta-lotes','Error al conectar con Flask.'); }
  finally  { setBtnLoading(btn, false); }
}

function renderLotes(data) {
  const tieneM   = !!data.metricas;
  const modeloNm = data.modelo_usado === 'LOGREG' ? 'Regresión Logística' : 'Red Neuronal';

  let html = `
    <div class="resumen-grid">
      <div class="resumen-card"><div class="resumen-num">${data.total}</div><div class="resumen-lbl">Total</div></div>
      <div class="resumen-card"><div class="resumen-num text-success-custom">${data.resumen.Benigno}</div><div class="resumen-lbl">Benignos</div></div>
      <div class="resumen-card"><div class="resumen-num text-danger-custom">${data.resumen.Maligno}</div><div class="resumen-lbl">Malignos</div></div>
    </div>`;

  if (tieneM) {
    const m  = data.metricas;
    const cm = m.confusion_matrix;
    html += `
      <div class="label-sm mb-2">Métricas de desempeño · ${modeloNm}</div>
      <div class="metricas-grid mb-4">
        ${mkMet('Accuracy',  m.accuracy,  'mc-acc','Predicciones correctas')}
        ${mkMet('Precisión', m.precision, 'mc-pre','Exactitud en predichos M')}
        ${mkMet('Recall',    m.recall,    'mc-rec','Malignos detectados')}
        ${mkMet('F1-Score',  m.f1_score,  'mc-f1', 'Balance precisión/recall')}
      </div>
      <div class="cm-section">
        <div class="cm-title">Matriz de Confusión</div>
        <table class="cm-table">
          <thead><tr><th></th><th>Pred. Benigno</th><th>Pred. Maligno</th></tr></thead>
          <tbody>
            <tr>
              <th style="font-size:.72rem;color:var(--text-faint);text-align:right;padding:6px 10px">Real Benigno</th>
              <td class="cm-tn"><span class="cm-val">${cm[0][0]}</span><span class="cm-tag">TN</span></td>
              <td class="cm-fp"><span class="cm-val">${cm[0][1]}</span><span class="cm-tag">FP</span></td>
            </tr>
            <tr>
              <th style="font-size:.72rem;color:var(--text-faint);text-align:right;padding:6px 10px">Real Maligno</th>
              <td class="cm-fn"><span class="cm-val">${cm[1][0]}</span><span class="cm-tag">FN</span></td>
              <td class="cm-tp"><span class="cm-val">${cm[1][1]}</span><span class="cm-tag">TP</span></td>
            </tr>
          </tbody>
        </table>
        <p style="font-size:.7rem;color:var(--text-faint);margin-top:6px">
          TN = Verdadero Negativo · FP = Falso Positivo · FN = Falso Negativo · TP = Verdadero Positivo
        </p>
      </div>`;
  } else {
    html += `<div class="aviso-sin-etiquetas"><i class="bi bi-info-circle me-1"></i>
      El CSV no incluye columna <strong>Diagnosis</strong>. Métricas no disponibles.</div>`;
  }

  html += `
    <div class="label-sm mb-2">Detalle por fila</div>
    <div class="detalle-table-wrap">
      <table class="detalle-table">
        <thead><tr><th>#</th><th>Predicción</th><th>P(Benigno)</th><th>P(Maligno)</th></tr></thead>
        <tbody>
          ${data.predicciones.map(r => `
            <tr>
              <td>${r.fila}</td>
              <td><span class="${r.clase===0?'pred-badge-b':'pred-badge-m'}">${r.prediccion}</span></td>
              <td>${r.prob_benigno}%</td>
              <td>${r.prob_maligno}%</td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>`;

  document.getElementById('resultado-lotes').innerHTML = html;
}

function mkMet(nom, val, cls, desc) {
  return `<div class="metrica-card ${cls}">
    <div class="metrica-val">${(val*100).toFixed(1)}%</div>
    <div class="metrica-nombre">${nom}</div>
    <div class="metrica-desc">${desc}</div>
  </div>`;
}

// ── Descargar CSV ejemplo ─────────────────────────
function descargarEjemplo() {
  if (!columnas.length) return;
  const hdr  = [...columnas,'Diagnosis'].join(',');
  const f1   = columnas.map(c => EJEMPLO[c] ?? '0').join(',') + ',M';
  const f2   = columnas.map(c => EJEMPLO[c] ? (EJEMPLO[c]*0.7).toFixed(4) : '0').join(',') + ',B';
  const blob = new Blob([`${hdr}\n${f1}\n${f2}`], {type:'text/csv'});
  const a    = Object.assign(document.createElement('a'), {href:URL.createObjectURL(blob), download:'ejemplo_breast_cancer.csv'});
  a.click(); URL.revokeObjectURL(a.href);
}

// ── Utilidades ────────────────────────────────────
function emptyState(icon, msg) {
  return `<div class="empty-state"><div class="empty-icon"><i class="bi bi-${icon}"></i></div><p>${msg}</p></div>`;
}

function showAlerta(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = `<div class="alert-custom mb-3"><i class="bi bi-exclamation-triangle"></i><span>${msg}</span></div>`;
  setTimeout(() => { el.innerHTML = ''; }, 5000);
}

function setBtnLoading(btn, on) {
  if (!btn) return;
  if (on) { btn.disabled=true; btn._t=btn.innerHTML; btn.innerHTML=`<span class="spinner-border spinner-border-sm me-2"></span>Procesando...`; }
  else    { btn.disabled=false; btn.innerHTML=btn._t; }
}