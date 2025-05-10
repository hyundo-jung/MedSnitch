import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import close from '../assets/close.png'
import './styles/LoginSignup.css';  

function SignupPage({ onClose }) {
  const [formData, setFormData] = useState({ username: '', password: '' });
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({...formData, [e.target.name]: e.target.value});
  };

  // const handleSubmit = async (e) => {
  //   e.preventDefault();
  //   try {
  //     const response = await fetch('http://localhost:8000/api/accounts/register/', {
  //       method: 'POST',
  //       headers: { 'Content-Type': 'application/json' },
  //       body: JSON.stringify(formData),
  //     });

  //     if (response.ok) {
  //       console.log('Signup successful');
  //       onClose();
  //     } else {
  //       console.error('Signup failed');
  //     }
  //   } catch (err) {
  //     console.error('Error:', err);
  //   }
  // };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Step 1: Register the user
      const registerResponse = await fetch('http://localhost:8000/api/accounts/register/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!registerResponse.ok) {
        console.error('Signup failed');
        return;
      }

      // Step 2: Log them in immediately
      const loginResponse = await fetch('http://localhost:8000/api/accounts/login/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(formData),
      });

      if (loginResponse.ok) {
        const result = await loginResponse.json();
        localStorage.setItem('username', result.username); // Optional: save username for navigation
        console.log('Signup + login successful');
        navigate(`/profile/${result.username}`);  // Go to profile
        onClose();  // Close modal if needed
      } else {
        console.error('Auto-login after signup failed');
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
