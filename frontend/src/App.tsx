import { FormEvent, useEffect, useRef, useState } from 'react';

type Model = { id: string; name: string; available: boolean };
type Result = { label: string; confidence: number; model: string; distribution: {label:string; probability:number}[]; currentDistribution:{label:string; probability:number}[]; contextMessages:number; contextualUpdated:boolean; contextualSummary?:string; final?:boolean; immediateRisk:boolean; recentRisk:boolean; assistantReply:string; replySource:'gemini'|'local-fallback'|'local-safety'; translatedText:string; sourceLanguage:'es'|'en'; disclaimer:string };
type Message = { id:number; role:'user'|'assistant'; text:string; result?:Result; error?:boolean; frame?:string|null };
type CompareItem = { model:string; name:string; available:boolean; label?:string; confidence?:number; distribution?:{label:string;probability:number}[] };

const fallbackModels: Model[] = [
  {id:'lightgbm', name:'LightGBM', available:true},
  {id:'logistic', name:'Regresión logística', available:false},
  {id:'svm', name:'SVM lineal', available:false},
  {id:'mental_roberta', name:'MentalRoBERTa', available:false},
];
const suggestions = [
  'Hoy me siento tranquilo y agradecido.',
  'Me siento vacío y agotado todos los días.',
  'Siento que las cosas han sido muy difíciles últimamente.',
];
const labelCopy: Record<string,{title:string;body:string;color:string}> = {
  Normal:{title:'No se detectó una señal predominante',body:'El patrón lingüístico se acerca más a la clase Normal.',color:'#6aaa8c'},
  Depression:{title:'Señales compatibles con depresión',body:'El texto contiene patrones que el modelo asocia con la clase Depression.',color:'#c4884a'},
  Suicidal:{title:'Señales de riesgo elevado',body:'El texto contiene patrones asociados con riesgo suicida. Si esto refleja cómo te sientes, busca ayuda humana inmediata.',color:'#bf554d'},
};

function ResultCard({result}:{result:Result}) {
  const copy = labelCopy[result.label] ?? labelCopy.Normal;
  return <div className="result-card" style={{'--accent':copy.color} as React.CSSProperties}>
    <div className="result-head"><span className="result-mark">✦</span><div><small>ANÁLISIS DEL MODELO</small><h3>{copy.title}</h3></div></div>
    <p>{copy.body}</p>
    <div className="bars">{result.distribution.map(item=><div className="bar-row" key={item.label}>
      <div><span>{item.label}</span><b>{Math.round(item.probability*100)}%</b></div>
      <div className="track"><i style={{width:`${item.probability*100}%`}} /></div>
    </div>)}</div>
    <div className="model-tag">Tendencia de {result.contextMessages} mensaje{result.contextMessages===1?'':'s'} · Modelo: {result.model} · Confianza: {Math.round(result.confidence*100)}% · Conversación: {result.replySource==='gemini'?'Gemini':'respuesta local'}</div>
    {result.sourceLanguage==='es'&&<details className="translation"><summary>Ver traducción analizada</summary><p>{result.translatedText}</p></details>}
    {(result.label==='Suicidal'||result.immediateRisk) && <div className="urgent"><strong>No estás solo/a.</strong> Contacta ahora a emergencias de tu país o a una persona de confianza. En Perú: Línea 113, opción 5.</div>}
    {result.label!=='Suicidal'&&!result.immediateRisk&&result.recentRisk&&<div className="recent-risk"><strong>Seguimiento recomendado.</strong> Aunque el mensaje actual muestra una mejoría, hubo una señal de riesgo reciente. Mantén contacto con una persona de confianza.</div>}
  </div>;
}

function TrendChart({results}:{results:Result[]}) {
  const recent=results.slice(-10);
  if(!recent.length)return null;
  const labels=['Normal','Depression','Suicidal'];
  const colors:Record<string,string>={Normal:'#6aaa8c',Depression:'#c4884a',Suicidal:'#bf554d'};
  const width=520,height=118,pad=10;
  const point=(label:string,index:number)=>{
    const value=recent[index].distribution.find(item=>item.label===label)?.probability??0;
    const x=recent.length===1?width/2:pad+index*(width-pad*2)/(recent.length-1);
    return {x,y:pad+(1-value)*(height-pad*2)};
  };
  return <section className="trend-panel" aria-label="Evolución de la tendencia lingüística">
    <div className="trend-title"><div><b>Evolución contextual</b><small>Últimos {recent.length} análisis</small></div><span>No representa severidad clínica</span></div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img">
      {[.25,.5,.75].map(value=><line key={value} x1={pad} x2={width-pad} y1={pad+(1-value)*(height-pad*2)} y2={pad+(1-value)*(height-pad*2)} />)}
      {labels.map(label=><polyline key={label} points={recent.map((_,index)=>{const p=point(label,index);return `${p.x},${p.y}`;}).join(' ')} fill="none" stroke={colors[label]} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />)}
      {labels.flatMap(label=>recent.map((_,index)=>{const p=point(label,index);return <circle key={`${label}-${index}`} cx={p.x} cy={p.y} r="3.5" fill={colors[label]}/>;}))}
    </svg>
    <div className="trend-legend">{labels.map(label=><span key={label}><i style={{background:colors[label]}}/>{label}</span>)}</div>
  </section>;
}

function ContextPanel({results}:{results:Result[]}){
  const contextual=results.filter(result=>result.contextualUpdated);
  const latest=contextual.at(-1);
  const latestMessage=results.at(-1);
  return <section className="context-dock" aria-label="Evaluación contextual fija">
    <div className="context-summary">
      <div className="context-heading"><div><b>Evaluación contextual</b><small>Se actualiza cada 4 respuestas y al finalizar</small></div>{latest&&<span>{latest.label}</span>}</div>
      {!latest?<p className="context-empty">Analizando cada mensaje. El primer resumen contextual aparecerá en la cuarta respuesta.</p>:<>
        <div className="context-bars">{latest.distribution.map(item=><div className="bar-row" key={item.label}><div><span>{item.label}</span><b>{Math.round(item.probability*100)}%</b></div><div className="track"><i style={{width:`${item.probability*100}%`,background:labelCopy[item.label]?.color??'#2d8065'}} /></div></div>)}</div>
        <div className="model-tag">{latest.final?'Clasificación final':'Resumen contextual'} · Modelo: {latest.model} · {latest.contextMessages} respuesta{latest.contextMessages===1?'':'s'}</div>
        {latest.contextualSummary&&<details className="translation"><summary>Ver resumen analizado</summary><p>{latest.contextualSummary}</p></details>}
      </>}
      {(latestMessage?.immediateRisk)&&<div className="urgent"><strong>Busca apoyo inmediato.</strong> Contacta a emergencias o a una persona de confianza. En Perú: Línea 113, opción 5.</div>}
      {!latestMessage?.immediateRisk&&latestMessage?.recentRisk&&<div className="recent-risk"><strong>Seguimiento recomendado.</strong> Hubo una señal de riesgo reciente; mantén contacto con una persona de confianza.</div>}
    </div>
    <TrendChart results={contextual}/>
  </section>;
}

export default function App(){
  const [models,setModels]=useState<Model[]>(fallbackModels);
  const [model,setModel]=useState('lightgbm');
  const [text,setText]=useState('');
  const [language,setLanguage]=useState<'es'|'en'>('es');
  const [loading,setLoading]=useState(false);
  const [compare,setCompare]=useState<CompareItem[]|null>(null);
  const [finalResult,setFinalResult]=useState<Result|null>(null);
  const [messages,setMessages]=useState<Message[]>([{id:1,role:'assistant',text:'Hola. Conversemos sobre cómo te has sentido. Si escribes en español, traduciré el texto al inglés antes de analizar la tendencia contextual, sin emitir diagnósticos.'}]);
  const bottom=useRef<HTMLDivElement>(null);
  const analysisResults=[...messages.flatMap(message=>message.result?[message.result]:[]),...(finalResult?[finalResult]:[])];

  useEffect(()=>{fetch('/api/models').then(r=>r.json()).then(setModels).catch(()=>{});},[]);
  useEffect(()=>bottom.current?.scrollIntoView({behavior:'smooth'}),[messages,loading]);

  async function submit(e?:FormEvent){
    e?.preventDefault(); const value=text.trim(); if(!value||loading)return;
    setFinalResult(null);
    setMessages(m=>[...m,{id:Date.now(),role:'user',text:value}]); setText(''); setLoading(true);
    const history=messages.slice(1).map(message=>({role:message.role,text:message.text,frame:message.frame})).slice(-29);
    setCompare(null);
    fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:value,history,language})})
      .then(r=>r.ok?r.json():null).then(data=>{if(data?.results)setCompare(data.results);}).catch(()=>{});
    try{
      const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:value,model,history,language})});
      const data=await response.json(); if(!response.ok)throw new Error(data.detail||'No se pudo analizar el texto.');
      setMessages(m=>[...m,{id:Date.now()+1,role:'assistant',text:data.assistantReply||'Este es el resultado del análisis:',result:data,frame:data.answerFrame}]);
    }catch(error){setMessages(m=>[...m,{id:Date.now()+1,role:'assistant',text:error instanceof Error?error.message:'Error inesperado.',error:true}]);}
    finally{setLoading(false)}
  }

  async function finalizeConversation(){
    if(loading||!messages.some(message=>message.role==='user'))return;
    setLoading(true);
    try{
      const history=messages.slice(1).map(message=>({role:message.role,text:message.text,frame:message.frame})).slice(-30);
      const response=await fetch('/api/finalize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model,history,language})});
      const data=await response.json(); if(!response.ok)throw new Error(data.detail||'No se pudo finalizar la conversación.');
      setFinalResult(data); setMessages(m=>m.slice(0,1)); setCompare(null);
    }catch(error){setMessages(m=>[...m,{id:Date.now(),role:'assistant',text:error instanceof Error?error.message:'Error inesperado.',error:true}]);}
    finally{setLoading(false)}
  }

  return <main>
    <header><a className="brand"><span>m</span><div><b>MHTC</b><small>MENTAL HEALTH TEXT CLASSIFIER</small></div></a><div className="status"><i/> Sistema experimental</div></header>
    <section className="hero"><div className="eyebrow">TECNOLOGÍA CON PROPÓSITO</div><h1>Un espacio para poner<br/><em>en palabras</em> lo que sientes.</h1><p>Analizamos texto con modelos de NLP para identificar señales tempranas y orientar hacia apoyo oportuno.</p></section>
    <section className="workspace">
      <aside><div className="aside-title">Modelo de análisis · último mensaje</div>{models.map(item=>{
        const verdict=compare?.find(entry=>entry.model===item.id);
        const color=verdict?.label?(labelCopy[verdict.label]?.color??'#8a8a8a'):undefined;
        return <button key={item.id} disabled={!item.available} onClick={()=>setModel(item.id)} className={model===item.id?'selected':''}>
          <span>{item.name}{model===item.id&&<b className="aside-active">activo</b>}</span>
          {verdict?.available?<>
            <small className="aside-verdict" style={{color}}>{verdict.label} · {Math.round((verdict.confidence??0)*100)}%</small>
            <span className="aside-stack">{verdict.distribution!.map(part=><i key={part.label} style={{width:`${part.probability*100}%`,background:labelCopy[part.label]?.color}}/>)}</span>
          </>:<small>{item.available?'Disponible':'Ejecuta el entrenamiento'}</small>}
        </button>;
      })}
        <div className="notice"><b>Importante</b><p>Esta herramienta es académica. No diagnostica ni reemplaza la evaluación de un profesional. La conversación se conserva solo durante la sesión; Gemini la usa para conversar y crear resúmenes factuales sin asignar categorías.</p></div>
      </aside>
      <div className="chat">
        <div className="chat-head"><div><span className="avatar">✦</span><div><b>Asistente MHTC</b><small>Análisis contextual · español o inglés</small></div></div>
          <div className="head-actions">
            <label className="model-toggle" title="Modelo que conduce la conversación">
              <span>Modelo</span>
              <select value={model} onChange={e=>setModel(e.target.value)}>
                {models.map(item=><option key={item.id} value={item.id} disabled={!item.available}>{item.name}{item.available?'':' (no disponible)'}</option>)}
              </select>
            </label>
            <button disabled={loading||!messages.some(message=>message.role==='user')} onClick={finalizeConversation}>Finalizar y limpiar</button>
          </div>
        </div>
        <ContextPanel results={analysisResults}/>
        <div className="messages">{messages.map(message=><div key={message.id} className={`message ${message.role}`}><div className="bubble">{message.text}</div></div>)}
          {loading&&<div className="message assistant"><div className="bubble typing"><i/><i/><i/> Analizando patrones...</div></div>}
          {messages.length===1&&<div className="suggestions">{suggestions.map(s=><button key={s} onClick={()=>setText(s)}>{s}</button>)}</div>}<div ref={bottom}/>
        </div>
        <form onSubmit={submit}><textarea value={text} onChange={e=>setText(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submit();}}} placeholder={language==='es'?'Escribe aquí cómo te sientes en español…':'Write here how you feel in English…'} maxLength={5000}/><div className="form-foot"><div><select value={language} onChange={e=>setLanguage(e.target.value as 'es'|'en')} aria-label="Idioma del texto"><option value="es">Español → Inglés</option><option value="en">Inglés</option></select><small>{text.length}/5000 · Enter para enviar</small></div><button disabled={!text.trim()||loading} aria-label="Enviar">↑</button></div></form>
      </div>
    </section>
    <footer>Proyecto académico · Universidad Peruana de Ciencias Aplicadas <span>Privacidad por diseño: el texto no se almacena</span></footer>
  </main>;
}
