import React from 'react'
import ReactDOM from 'react-dom/client'
import { Provider } from 'react-redux'
import { createStore, combineReducers, applyMiddleware } from 'redux'
import keplerGlReducer from '@kepler.gl/reducers'
import { taskMiddleware } from 'react-palm/tasks'
import App from './App'
import './index.css'

const reducers = combineReducers({
  keplerGl: keplerGlReducer.initialState({
    uiState: {
      activeSidePanel: null,
      currentModal: null,
      readOnly: true,
    },
  }),
})

const store = createStore(reducers, {}, applyMiddleware(taskMiddleware))
window.__redux_store = store  // добавь эту строку
ReactDOM.createRoot(document.getElementById('root')).render(
  <Provider store={store}>
    <App />
  </Provider>
)
