import { useNavigate } from 'react-router-dom';

function AboutUs() {
    const navigate = useNavigate(); 

    return (
        <div>
            <button onClick={() => navigate('/')}>Back</button>
            <p>hello </p>
        </div>
    )
}
export default AboutUs