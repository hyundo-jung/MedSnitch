import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import close from '../assets/close.png'
import './styles/LoginSignup.css'; 

function LoginPage({ onClose }) {
  const [formData, setFormData] = useState({ username: '', password: '' });
  const navigate = useNavigate(); 

  const handleChange = (e) => {
    setFormData({...formData, [e.target.name]: e.target.value});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/api/accounts/login/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',  // <- Needed to include cookies
        body: JSON.stringify(formData),
      });
  
      if (response.ok) {
        const result = await response.json();
        localStorage.setItem('username', result.username);
        console.log('Login successful');
        navigate(`/profile/${result.username}`);  // Redirect to profile page
      } else {
        console.error('Login failed');
      }
    } catch (err) {
      console.error('Error:', err);
    }
  };
  

  return (
    <div className="popup-backdrop">
      <div className="popup">
        <img className="close-popup" src={close} onClick={onClose}></img>
        <h2>Log In</h2>
        <form onSubmit={handleSubmit}>
          <input name="username" placeholder="Username" onChange={handleChange} required /><br />
          <input name="password" type="password" placeholder="Password" onChange={handleChange} required /><br />
          <button type="submit">Log In</button>
        </form>
      </div>
    </div>
  );
}

export default LoginPage;
