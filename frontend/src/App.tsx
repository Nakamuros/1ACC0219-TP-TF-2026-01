import { FormEvent, useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import {
  IconBrain, IconChartBar,
  IconSend, IconRefresh, IconAlertTriangle,
  IconTrendingUp, IconHeartHandshake,
} from '@tabler/icons-react';
import './styles.css';

type Model  = { id: string; name: string; available: boolean };
type Dist   = { label: string; probability: number };
type Result = {
  label: string; confidence: number; model: string;
  distribution: Dist[]; final?: boolean;
  immediateRisk: boolean; assistantReply: string;
  translatedText: string; sourceLanguage: 'es' | 'en';
  disclaimer: string;
};
type Msg = { id: number; role: 'user' | 'assistant'; text: string; result?: Result; err?: boolean };

const FALLBACK_MODELS: Model[] = [
  { id: 'lightgbm',      name: 'LightGBM',           available: true  },
  { id: 'logistic',      name: 'Reg. Logística',      available: false },
  { id: 'svm',           name: 'SVM Lineal',          available: false },
  { id: 'mental_roberta',name: 'MentalRoBERTa',       available: false },
];

const SUGGESTIONS = [
  'Hoy me siento tranquilo y agradecido.',
  'Me siento vacío y agotado todos los días.',
  'Siento que las cosas han sido muy difíciles últimamente.',
];

const SEG_COLORS: Record<string, string> = { Normal: '#6aaa8c', Depression: '#c4884a', Suicidal: '#bf554d' };
const SEG_CLASS:  Record<string, string> = { Normal: 'n', Depression: 'd', Suicidal: 's' };

/* ── Spline helper ── */
function spline(pts: {x:number;y:number}[]): string {
  if (pts.length < 2) return pts.map(p=>`${p.x},${p.y}`).join(' ');
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const cp1x = pts[i].x   + (pts[i+1].x - (pts[i-1]?.x ?? pts[i].x)) / 5;
    const cp1y = pts[i].y   + (pts[i+1].y - (pts[i-1]?.y ?? pts[i].y)) / 5;
    const cp2x = pts[i+1].x - (pts[i+2]?.x ?? pts[i+1].x - pts[i].x)  / 5;
    const cp2y = pts[i+1].y - (pts[i+2]?.y ?? pts[i+1].y - pts[i].y)  / 5;
    d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${pts[i+1].x} ${pts[i+1].y}`;
  }
  return d;
}

/* ── ResultCard ── */
function ResultCard({ result }: { result: Result }) {
  const [showTrans, setShowTrans] = useState(false);
  return (
    <div className="res">
      <div className="res-lbl">{result.label}</div>
      <div className="res-bars">
        {result.distribution.map(d => (
          <div className="res-row" key={d.label}>
            <div className="res-meta"><span>{d.label}</span><b>{Math.round(d.probability * 100)}%</b></div>
            <div className="res-track"><div className="res-fill" style={{ width: `${d.probability * 100}%` }} /></div>
          </div>
        ))}
      </div>
      <div className="res-foot">Modelo: {result.model} · Confianza: {Math.round(result.confidence * 100)}%</div>
      {result.sourceLanguage === 'es' && (
        <details className="res-trans" open={showTrans} onToggle={e => setShowTrans((e.target as HTMLDetailsElement).open)}>
          <summary>Ver traducción analizada</summary>
          <p>{result.translatedText}</p>
        </details>
      )}
      {(result.label === 'Suicidal' || result.immediateRisk) && (
        <div className="res-urgent">⚠ No estás solo/a. En Perú: Línea 113, opción 5.</div>
      )}
    </div>
  );
}

/* ── TrendWidget ── */
function TrendWidget({ results }: { results: Result[] }) {
  const recent = results.slice(-10);
  const labels  = ['Normal', 'Depression', 'Suicidal'];
  const W = 260, H = 100, P = 10;

  if (!recent.length) return <div className="trend-empty">Aparecerá con el primer análisis.</div>;

  const pts = (label: string) => recent.map((r, i) => {
    const val = r.distribution.find(d => d.label === label)?.probability ?? 0;
    const x   = recent.length === 1 ? W / 2 : P + i * (W - P * 2) / (recent.length - 1);
    const y   = P + (1 - val) * (H - P * 2);
    return { x, y };
  });

  return (
    <>
      <div className="trend-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} height={H}>
          {[.25, .5, .75].map(v => (
            <line key={v} x1={P} x2={W - P}
              y1={P + (1 - v) * (H - P * 2)} y2={P + (1 - v) * (H - P * 2)}
              stroke="#EAE0C8" strokeWidth="1" strokeDasharray="3 4" />
          ))}
          {labels.map(l => (
            <path key={l} d={spline(pts(l))}
              fill="none" stroke={SEG_COLORS[l]} strokeWidth="2.5"
              strokeLinecap="round" strokeLinejoin="round" />
          ))}
          {labels.flatMap(l => pts(l).map((p, i) => (
            <circle key={`${l}-${i}`} cx={p.x} cy={p.y} r="3.5" fill={SEG_COLORS[l]} />
          )))}
        </svg>
      </div>
      <div className="trend-leg">
        {labels.map(l => (
          <span key={l}><i style={{ background: SEG_COLORS[l] }} />{l}</span>
        ))}
      </div>
    </>
  );
}

/* ── AnalysisWidget ── */
function AnalysisWidget({ result }: { result: Result | null }) {
  if (!result) return <div className="w-empty">Envía un mensaje para ver el análisis.</div>;
  const dist = result.distribution;
  return (
    <>
      <div className="w-label" style={{ color: SEG_COLORS[result.label] ?? '#4a4035' }}>{result.label}</div>
      <div className="w-conf">Confianza: {Math.round(result.confidence * 100)}% · Modelo: {result.model}</div>
      <div className="prop-bar">
        {dist.map(d => (
          <div key={d.label} className={`seg ${SEG_CLASS[d.label]}`}
            style={{ width: `${d.probability * 100}%` }} />
        ))}
      </div>
      <div className="leg">
        {dist.map(d => (
          <div className="leg-it" key={d.label}>
            <div className="leg-dot" style={{ background: SEG_COLORS[d.label] }} />
            {d.label} {Math.round(d.probability * 100)}%
          </div>
        ))}
      </div>
    </>
  );
}

/* ── Main App ── */
export default function App() {
  const [models,   setModels]   = useState<Model[]>(FALLBACK_MODELS);
  const [model,    setModel]    = useState('lightgbm');
  const [lang,     setLang]     = useState<'es'|'en'>('es');
  const [text,     setText]     = useState('');
  const [loading,  setLoading]  = useState(false);
  const [messages, setMessages] = useState<Msg[]>([
    { id: 1, role: 'assistant', text: 'Hola. Conversemos sobre cómo te has sentido. Puedes escribir en español o inglés.' },
  ]);

  const bottom = useRef<HTMLDivElement>(null);
  const allResults = messages.flatMap(m => m.result ? [m.result] : []);
  const latestResult = allResults.at(-1) ?? null;

  useEffect(() => { fetch('/api/models').then(r => r.json()).then(setModels).catch(() => {}); }, []);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);

  /* GSAP pop on new message */
  const popRef = useRef<HTMLDivElement[]>([]);
  useEffect(() => {
    const el = popRef.current.at(-1);
    if (el) gsap.from(el, { scale: 0.82, y: 8, opacity: 0, duration: 0.38, ease: 'back.out(1.7)' });
  }, [messages.length]);

  async function submit(e?: FormEvent) {
    e?.preventDefault();
    const val = text.trim();
    if (!val || loading) return;
    const userMsg: Msg = { id: Date.now(), role: 'user', text: val };
    setMessages(m => [...m, userMsg]);
    setText('');
    setLoading(true);
    try {
      const history = messages.slice(1).map(m => ({ role: m.role, text: m.text })).slice(-29);
      const res  = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: val, model, history, language: lang }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'No se pudo analizar el texto.');
      setMessages(m => [...m, { id: Date.now() + 1, role: 'assistant', text: data.assistantReply ?? 'Análisis completado.', result: data }]);
    } catch (err) {
      setMessages(m => [...m, { id: Date.now() + 1, role: 'assistant', text: err instanceof Error ? err.message : 'Error inesperado.', err: true }]);
    } finally {
      setLoading(false);
    }
  }

  async function finalize() {
    if (loading || !messages.some(m => m.role === 'user')) return;
    setLoading(true);
    try {
      const history = messages.slice(1).map(m => ({ role: m.role, text: m.text })).slice(-30);
      const res  = await fetch('/api/finalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, history, language: lang }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Error al finalizar.');
      setMessages([
        { id: 1, role: 'assistant', text: 'Hola. Conversemos sobre cómo te has sentido. Puedes escribir en español o inglés.' },
        { id: Date.now(), role: 'assistant', text: `Clasificación final: ${data.label}`, result: { ...data, final: true } },
      ]);
    } catch (err) {
      setMessages(m => [...m, { id: Date.now(), role: 'assistant', text: err instanceof Error ? err.message : 'Error inesperado.', err: true }]);
    } finally {
      setLoading(false);
    }
  }

  const showAlert = latestResult?.immediateRisk || latestResult?.label === 'Suicidal';

  return (
    <div className="app">

      {/* ── Rail ── */}
      <nav className="rail">
        <div className="rail-logo">m</div>
        <div className="rail-sep" />
        <button className="rail-btn active" title="Análisis"><IconBrain size={20} /></button>
      </nav>

      {/* ── Center ── */}
      <main className="center">
        {/* header */}
        <div className="chat-hdr">
          <div className="chat-hdr-l">
            <div className="chat-avatar"><IconBrain size={22} /></div>
            <div>
              <div className="chat-hdr-title">Asistente MHTC</div>
              <div className="chat-hdr-sub">Análisis contextual · español o inglés</div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="hdr-btn" onClick={finalize}
              disabled={loading || !messages.some(m => m.role === 'user')}>
              <IconRefresh size={14} /> Finalizar
            </button>
          </div>
        </div>

        {/* model chips */}
        <div className="chips">
          {models.filter(m => m.available).map(m => (
            <button key={m.id} className={`chip${model === m.id ? ' on' : ''}`}
              onClick={() => setModel(m.id)}>
              {m.name}
            </button>
          ))}
        </div>

        {/* messages */}
        <div className="msgs">
          {messages.map((m, i) => (
            <div key={m.id} className={`msg ${m.role}${m.err ? ' err' : ''}`}
              ref={el => { if (el) popRef.current[i] = el; }}>
              <div className="bbl">
                {m.text}
                {m.result && <ResultCard result={m.result} />}
              </div>
            </div>
          ))}
          {loading && (
            <div className="msg assistant">
              <div className="bbl">
                <div className="typing-bbl">
                  <div className="dot" /><div className="dot" /><div className="dot" />
                </div>
              </div>
            </div>
          )}
          {messages.length === 1 && !loading && (
            <div className="sugg">
              {SUGGESTIONS.map(s => (
                <button key={s} className="sugg-btn" onClick={() => setText(s)}>{s}</button>
              ))}
            </div>
          )}
          <div ref={bottom} />
        </div>

        {/* input */}
        <div className="inp-area">
          <form onSubmit={submit}>
            <div className="inp-shell">
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }}
                placeholder={lang === 'es' ? 'Escribe cómo te sientes en español…' : 'Write how you feel in English…'}
                maxLength={5000}
                rows={1}
              />
              <div className="inp-right">
                <span className="char">{text.length}/5000</span>
                <button type="button" className={`lang-btn${lang === 'es' ? ' on' : ''}`}
                  onClick={() => setLang(l => l === 'es' ? 'en' : 'es')}>
                  {lang === 'es' ? 'ES→EN' : 'EN'}
                </button>
                <button type="submit" className="send" disabled={!text.trim() || loading}>
                  <IconSend size={17} />
                </button>
              </div>
            </div>
          </form>
        </div>

        <div className="disclaimer">Proyecto académico UPC · no constituye diagnóstico clínico · texto no almacenado</div>
      </main>

      {/* ── Right Panel ── */}
      <aside className="panel">
        {/* Analysis */}
        <div className="widget">
          <div className="w-title"><IconChartBar size={15} /> Análisis actual</div>
          <AnalysisWidget result={latestResult} />
        </div>

        {/* Trend */}
        <div className="widget">
          <div className="w-title"><IconTrendingUp size={15} /> Tendencia</div>
          <TrendWidget results={allResults} />
        </div>

        {/* Alert */}
        <div className="widget alert-widget">
          <div className="w-title"><IconHeartHandshake size={15} /> Recursos de apoyo</div>
          {showAlert && (
            <div className="alert-urgent"><IconAlertTriangle size={14} /> Busca apoyo inmediato</div>
          )}
          <p className="alert-line"><strong>Perú:</strong> Línea 113, opción 5</p>
          <p className="alert-line"><strong>Argentina:</strong> Centro de Asistencia al Suicida · 135</p>
          <p className="alert-line"><strong>México:</strong> SAPTEL · 55 5259-8121</p>
          <p className="alert-line"><strong>España:</strong> Teléfono de la Esperanza · 717 003 717</p>
        </div>
      </aside>

    </div>
  );
}
