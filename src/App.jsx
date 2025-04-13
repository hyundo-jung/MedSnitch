import { useState } from 'react'
import { useNavigate } from 'react'
import { BrowserRouter as Router, Route, Routes,} from 'react-router-dom';
import logo from './assets/Logo.png'
import arrow from './assets/arrow-down.png'
import './App.css'
import HomePage from './pages/HomePage'
import AboutUs from './pages/info pages/AboutUs'

function App() {
  
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage/>} />
        <Route path="/about-us" element={<AboutUs />} />
      </Routes>
    </Router>
  )
}

export default App
