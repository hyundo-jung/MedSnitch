import { useState } from 'react'
import logo from '../assets/Logo.png'
import arrow from '../assets/arrow-down.png'
import './styles/HomePage.css'
import InfoWidget from './components/InfoWidget'
import Circle from './components/Circle'
import arrow1 from '../assets/arrow1.png'
import arrow2 from '../assets/arrow2.png'
import arrow3 from '../assets/arrow3.png'
import arrow4 from '../assets/arrow4.png'
import arrow5 from '../assets/arrow5.png'
import lock from '../assets/lock.png'

import LoginPage from './LoginPage';
import SignupPage from './SignupPage';


function HomePage() {

  const [showLogin, setShowLogin] = useState(false);
  const [showSignup, setShowSignup] = useState(false);

  const openLogin = () => setShowLogin(true);
  const openSignup = () => setShowSignup(true);

  const closeLogin = () => setShowLogin(false);
  const closeSignup = () => setShowSignup(false);
 

  const data = [
    {title: "What is Medical Bill Fraud?", 
    blurb: "bill fraud blurb",
    link: "https://www.google.com/"}, 
    {title: "Training Data Overview", 
    blurb: "source, summary",
    link: "https://www.google.com/"},
    {title: "Meet the Team", 
    blurb: "IEEE intro",
    link: "https://www.google.com/"}]; 
  
  return (
    <>
        <div className="landing">
          <div className="title">
              <h1 className="motto">Healthcare should not have hidden costs. </h1>
              <h2 className="intro">
                <span className="highlight">MedSnitch</span> detects fraudulent medical billing with a click of a button.
              </h2>

          </div>
          <div className="logo">
              <img src={logo}></img>
          </div>
        </div>   
        
        <div className="scroll">
            <h3>How it works</h3>
            <img className="scroll-arrow" src={arrow}></img>
        </div>

        <div className="buttons">
          <button className="login-button" onClick={openLogin}>
            Log in 
          </button>
          <button className="login-button" onClick={openSignup}>
            Sign Up 
          </button>
        </div>
        {showLogin && <LoginPage onClose={closeLogin} />}
        {showSignup && <SignupPage onClose={closeSignup} />}

        <div className="flowchart-circles">
          <Circle num="1" color="#B6FFDA" text="Log In" top="60px" left="50%" /> 
          <Circle num="2" color="#BEDAF7" text="Upload Claims" top="345px" left="82%"/>
          <img src={arrow1} className="arrow1"></img>
          <img src={arrow2} className="arrow2"></img>
          <img src={arrow3} className="arrow3"></img>
          <img src={arrow4} className="arrow4"></img>
          <img src={arrow5} className="arrow5"></img>
          
          <Circle num="3" color="#E4BEF7" text="Test with Our Model" top="775px" left="70%"/>
          <Circle num="4" color="#BEDAF7" text="Get Fraud Statistics" top="775px" left="30%"/>
          <Circle num="5" color="#E4BEF7" text="Save History"  top="345px" left="18%"/>
          <p className="flowchart-text">Keep all of your tested claims and results in one place.</p>
          
        </div>
         
  
        <div className="learn-more-container">
          <p className="learn-more">Learn More</p>
          <hr className="learn-more" style={{width: "25%"}}/>
          <div className="widgets">
              <InfoWidget {...data[0]}/>
              <InfoWidget {...data[1]}/>
              <InfoWidget {...data[2]}/>
          </div>
        </div>
        <div className="footer">
          <img src={lock} className="lock"></img>
          <p className="contact">Contact Us: email/number</p>
        </div>

        
       {/* <p style={{fontFamily:"mono", fontSize: "25px", position: "absolute", left: "50%", transform: "translate(-50%)", top: "3100px"}}>Contact Us: email/number</p>
         */}

        </>
      )
    }

export default HomePage