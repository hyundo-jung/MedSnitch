// used for flowchart in homepage
import '../styles/Circle.css'

function Circle({ num, color, text, top, left }) {
    return (
        <div className="circle" style={{backgroundColor: color, position: 'absolute', top: top, left: left, transform: 'translate(-50%, -50%)'}}>
            <h1 className="circle-num">{num}</h1>
            <h2 className="circle-text">{text}</h2>
        </div>
    )
}
export default Circle