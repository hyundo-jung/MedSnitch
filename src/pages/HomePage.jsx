import { useState } from 'react'
import logo from '../assets/Logo.png'
import arrow from '../assets/arrow-down.png'
import './styles/HomePage.css'
import InfoWidget from './components/InfoWidget'
import Circle from './components/Circle'

function HomePage() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)

  const data = {
    title: "What is this data?", 
    blurb: "lalalalalaalalal",
    link: "https://www.google.com/", 
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

        <div className="flowchart">
          <Circle num="1" color="#B6FFDA" text="Log In" top="100px" left="50%"/> 

          <Circle num="2" color="#BEDAF7" text="Upload Claims" top="375px" left="80%"/>
          <div className="arrow">
            <svg xmlns="http://www.w3.org/2000/svg" width="114" height="95" viewBox="0 0 114 95" fill="none">
              <path d="M111.069 92.2888C112.441 92.1363 113.43 90.9003 113.278 89.528L110.793 67.1656C110.64 65.7933 109.404 64.8045 108.032 64.957C106.66 65.1094 105.671 66.3455 105.823 67.7178L108.032 87.5954L88.1544 89.8041C86.7822 89.9565 85.7933 91.1926 85.9458 92.5648C86.0983 93.9371 87.3343 94.926 88.7066 94.7735L111.069 92.2888ZM0.15921 4.49868L109.231 91.7562L112.355 87.8519L3.28268 0.594337L0.15921 4.49868Z" fill="white"/>
            </svg>
          </div>
          
          <Circle num="3" color="#E4BEF7" text="Test with Our Model" top="825px" left="70%"/>
          <div className="arrow">
            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="135" viewBox="0 0 64 135" fill="none">
              <path d="M10.2845 133.239C10.8585 134.495 12.3418 135.048 13.5976 134.474L34.0607 125.119C35.3164 124.545 35.8691 123.062 35.295 121.806C34.721 120.55 33.2377 119.998 31.9819 120.572L13.7924 128.887L5.47724 110.697C4.9032 109.442 3.41988 108.889 2.16416 109.463C0.908436 110.037 0.355825 111.52 0.92987 112.776L10.2845 133.239ZM58.9791 0.435333L10.2154 131.327L14.9009 133.073L63.6645 2.18088L58.9791 0.435333Z" fill="white"/>
            </svg>
          </div>
          <Circle num="4" color="#BEDAF7" text="Get Fraud Statistics" top="825px" left="30%"/>
          <Circle num="5" color="#E4BEF7" text="Save History"  top="375px" left="20%"/>

          <p className="flowchart-text">Keep all of your tested claims and results in one place.</p>
        </div>

        <div className="widgets">
            <InfoWidget {...data}/>
            <InfoWidget {...data}/>
            <InfoWidget {...data}/>
        </div>

    </>
  )
}

export default HomePage