import { useState } from 'react'
import logo from '../assets/Logo.png'
import arrow from '../assets/arrow-down.png'
import './styles/HomePage.css'
import InfoWidget from './components/InfoWidget'

function HomePage() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)

  const data = {
    title: "What is this data?", 
    blurb: "lalalalalaalalal",
    link: "https://www.google.com/"
  }; 
  
  return (
    <>
        <div className="landing">
        <div className="title">
            <h1 className="motto">Healthcare should not have hidden costs. </h1>
            <h2 className="intro">MedSnitch detects fraudulent medical billing with a click of a button.  </h2>
        </div>
        <div className="logo">
            <img src={logo}></img>
        </div>
        </div>   
        
        <div className="scroll">
            <h3>How it works</h3>
            <img src={arrow}></img>
        </div>

        <button className="login-button" onClick={() => setIsLoggedIn((isLoggedIn) => !isLoggedIn)}>
            Log In {isLoggedIn}
        </button>

        <div className="widgets">
            <InfoWidget {...data}/>
            <InfoWidget {...data}/>
            <InfoWidget {...data}/>
        </div>

    </>
  )
}

export default HomePage