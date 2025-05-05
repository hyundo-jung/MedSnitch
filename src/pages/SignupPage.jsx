import { useState } from 'react';
import close from '../assets/close.png'
import './styles/LoginSignup.css';  

function SignupPage({ onClose }) {
  const [formData, setFormData] = useState({ username: '', password: '' });

  const handleChange = (e) => {
    setFormData({...formData, [e.target.name]: e.target.value});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/api/accounts/register/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        console.log('Signup successful');
        onClose();
      } else {
        console.error('Signup failed');
      }
    } catch (err) {
      console.error('Error:', err);
    }
  };

  return (
    <div className="popup-backdrop">
      <div className="popup">
        <img className="close-popup" src={close} onClick={onClose}></img>
        <h2>Create An Account</h2>
        <form onSubmit={handleSubmit}>
          <input name="username" placeholder="Username" onChange={handleChange} required /><br />
          <input name="password" type="password" placeholder="Password" onChange={handleChange} required /><br />
          <button type="submit">Sign Up</button>
        </form>
      </div>
    </div>
  );
}

export default SignupPage;
