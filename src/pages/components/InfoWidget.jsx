// homepage infowidget 
import { useState } from 'react'
import { useNavigate } from 'react-router-dom';
import '../styles/InfoWidget.css' 

function InfoWidget({title, blurb, link}) {
    const [isClicked, setIsClicked] = useState(false); 
    const navigate = useNavigate(); 

    return (
        <div className={`widget-container ${isClicked ? 'clicked' : ''}`}>
            <div className={`widget-title ${isClicked ? 'clicked' : ''}`} onClick={() => setIsClicked(!isClicked)}>
                {title}
            </div>
         
            {isClicked && 
            <>
                <p className="widget-blurb">
                    {blurb}
                    <a 
                        className={`read-more ${isClicked ? 'clicked' : ''}`} 
                        href={link} 
                        target="_blank" 
                        rel="noopener noreferrer"
                    >
                    Read more {'>'}
                    </a>

                </p>
               
            </>
            }
        </div>

    )
}
export default InfoWidget