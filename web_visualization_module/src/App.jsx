import React, { useEffect, useRef, useState } from 'react'
import KeplerGl from '@kepler.gl/components'
import { processGeojson } from '@kepler.gl/processors'
import { useDispatch } from 'react-redux'
import AutoSizer from 'react-virtualized/dist/commonjs/AutoSizer'
import { KEPLER_CONFIG } from './keplerConfig'
import './App.css'
import { addDataToMap, layerConfigChange, wrapTo } from '@kepler.gl/actions'

const MAPBOX_TOKEN = import.meta.env.VITE_MAPTILER_KEY ?? ''

const TEAM_MEMBERS = [
  'Илья Коротаев',
  'Бондаренко Григорий',
  'Григорьев Глеб',
  'Чиликин Александр',
  'Шкадин Роман',
]

function useCountUp(target, duration = 1800, started = false) {
  const [value, setValue] = useState(0)
  useEffect(() => {
    if (!started) return
    let start = null
    const step = (ts) => {
      if (!start) start = ts
      const progress = Math.min((ts - start) / duration, 1)
      const ease = 1 - Math.pow(1 - progress, 3)
      setValue(Math.floor(ease * target))
      if (progress < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }, [target, duration, started])
  return value
}

function StatCard({ value, suffix, label, delay, started }) {
  const count = useCountUp(value, 1600, started)
  return (
    <div className="stat-card" style={{ animationDelay: `${delay}ms` }}>
      <div className="stat-value">
        {count.toLocaleString('ru-RU')}
        <span className="stat-suffix">{suffix}</span>
      </div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function Hero({ onScrollToMap }) {
  const [statsStarted, setStatsStarted] = useState(false)
  const statsRef = useRef(null)

  useEffect(() => {
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setStatsStarted(true) },
      { threshold: 0.3 }
    )
    if (statsRef.current) obs.observe(statsRef.current)
    return () => obs.disconnect()
  }, [])

  return (
    <section className="hero">
      <div className="hero-grid" aria-hidden="true" />
      <div className="hero-rays" aria-hidden="true">
        <div className="ray ray-1" />
        <div className="ray ray-2" />
        <div className="ray ray-3" />
      </div>

      <header className="hero-header">
        <div className="logos">
          <MtsLogo />
          <span className="logo-divider" />
          <ChangellengeWordmark />
        </div>
        <div className="header-badge">
          <p>Changellenge &gt;&gt; Cup IT 2026</p>
        </div>
      </header>

      <div className="hero-content">
        <div className="hero-eyebrow">
          <span className="eyebrow-dot" />
          МТС True Tech
          <span className="eyebrow-line" />
          Команда: "Пока без названия"
        </div>

        <h1 className="hero-title">
          <span className="title-line title-line-1">Выше</span>
          <span className="title-line title-line-2">
            <span className="title-accent">крыши</span>
          </span>
          <span className="title-sub">Единая модель высотности зданий СПб</span>
        </h1>

        <div className="stats-row" ref={statsRef}>
          <StatCard value={155439} suffix="" label="зданий" delay={0} started={statsStarted} />
          <StatCard value={99} suffix=" м" label="макс. высота" delay={100} started={statsStarted} />
          <StatCard value={12.69} suffix=" м" label="средняя высота" delay={200} started={statsStarted} />
        </div>

        <button className="cta-button" onClick={onScrollToMap}>
          <span>Смотреть карту</span>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M12 5v14M5 12l7 7 7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>

      <div className="team-strip">
        <span className="team-label">Команда:</span>
        {TEAM_MEMBERS.map((name, i) => (
          <span key={i} className="team-name">
            {name}
            {i < TEAM_MEMBERS.length - 1 && <span className="team-sep">·</span>}
          </span>
        ))}
      </div>

      <div className="scroll-hint" onClick={onScrollToMap}>
        <div className="scroll-line" />
        <span>прокрутите вниз</span>
      </div>
    </section>
  )
}

const LAYERS = [
  { id: 'height', label: 'Высота', icon: '▲', active: true },
  { id: 'risk', label: 'Зоны покрытия', icon: '⚠', active: true },
]

function LayerPanel({ activeLayer, onSwitch }) {
  const dispatch = useDispatch()

  const handleSwitch = (id) => {
    onSwitch(id)
    const layers = window.__redux_store.getState().keplerGl['spb-map'].visState.layers
    const buildingLayer = layers.find(l => l.id === 'buildings-3d')
    const zonesLayer = layers.find(l => l.id === 'zones-layer')

    if (id === 'height') {
      dispatch(wrapTo('spb-map', layerConfigChange(buildingLayer, { isVisible: true })))
      dispatch(wrapTo('spb-map', layerConfigChange(zonesLayer, { isVisible: false })))
    }
    if (id === 'risk') {
      dispatch(wrapTo('spb-map', layerConfigChange(buildingLayer, { isVisible: false })))
      dispatch(wrapTo('spb-map', layerConfigChange(zonesLayer, { isVisible: true })))
    }
  }

  return (
    <div className="layer-panel">
      <div className="layer-panel-title">Слои</div>
      {LAYERS.map((l) => (
        <button
          key={l.id}
          className={`layer-btn ${activeLayer === l.id ? 'active' : ''} ${!l.active ? 'disabled' : ''}`}
          onClick={() => l.active && handleSwitch(l.id)}
          title={!l.active ? 'Скоро' : l.label}
        >
          <span className="layer-icon">{l.icon}</span>
          <span className="layer-label">{l.label}</span>
          {!l.active && <span className="layer-soon">soon</span>}
        </button>
      ))}
    </div>
  )
}

function MapLegend() {
  return (
    <div className="map-legend">
      <div className="legend-title">Высота зданий</div>
      <div className="legend-gradient" />
      <div className="legend-labels">
        <span>0 м</span>
        <span>50 м</span>
        <span>88+ м</span>
      </div>
    </div>
  )
}

function MapSection({ mapRef }) {
  const dispatch = useDispatch()
  const [loaded, setLoaded] = useState(false)
  const [activeLayer, setActiveLayer] = useState('height')

  useEffect(() => {
    Promise.all([
      fetch('/output.geojson').then(r => r.json()),
      fetch('/zones.geojson').then(r => r.json()),
    ]).then(([geojson1, geojson2]) => {
      const buildings = processGeojson(geojson1)
      const zones = processGeojson(geojson2)
      dispatch(addDataToMap({
        datasets: [
          { info: { label: 'Здания СПб', id: 'spb-buildings' }, data: buildings },
          { info: { label: 'Зоны МТС', id: 'mts-zones' }, data: zones },
        ],
        option: { centerMap: true, readOnly: true },
        config: KEPLER_CONFIG.config,
      }))
      setLoaded(true)
    }).catch(console.error)
  }, [dispatch])

  return (
    <section className="map-section" ref={mapRef}>
      <div className="map-header">
        <div className="map-header-left">
          <div className="section-eyebrow">
            <span className="eyebrow-dot" />
            Интерактивная карта
          </div>
          <h2 className="section-title">
            3D-модель высотности <span className="title-accent">Санкт-Петербурга</span>
          </h2>
        </div>
        <div className="map-header-right">
          <div className="map-hint">
            <span className="hint-icon">🖱</span> Зажмите Ctrl + перетащите для поворота камеры
          </div>
        </div>
      </div>

      <div className="map-wrapper">
        <LayerPanel activeLayer={activeLayer} onSwitch={setActiveLayer} />
        <MapLegend />
        {!loaded && (
          <div className="map-loading">
            <div className="loading-spinner" />
            <span>Загрузка данных…</span>
          </div>
        )}
        <AutoSizer>
          {({ height, width }) => (
            <KeplerGl
              id="spb-map"
              mapboxApiAccessToken={MAPBOX_TOKEN}
              width={width}
              height={height}
            />
          )}
        </AutoSizer>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-left">
          <MtsLogo small />
          <span className="footer-text">
            Changellenge &gt;&gt; Cup IT 2026 — Задача «Выше крыши»
          </span>
        </div>
        <div className="footer-right">
          <span className="footer-text">
            Пока без названия
          </span>
        </div>
      </div>
    </footer>
  )
}

function MtsLogo({ small }) {
  return (
    <img
      src="/mtslogo.svg"
      alt="МТС"
      width={small ? 55 : 70}
      height={small ? 25 : 31}
      style={{ objectFit: 'contain' }}
    />
  )
}

function ChangellengeWordmark() {
  return (
    <span className="changellenge-wordmark">
      <span className="cw-main">Changellenge</span>
      <span className="cw-cup">Cup IT</span>
    </span>
  )
}

export default function App() {
  const mapRef = useRef(null)

  const scrollToMap = () => {
    mapRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="app">
      <Hero onScrollToMap={scrollToMap} />
      <MapSection mapRef={mapRef} />
      <Footer />
    </div>
  )
}