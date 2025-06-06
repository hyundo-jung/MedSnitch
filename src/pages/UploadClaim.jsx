// src/pages/UploadClaim.jsx

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './styles/UploadClaim.css';  // your CSS

// helper to read Django's CSRF token cookie
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    document.cookie.split(';').forEach(c => {
      const [k, v] = c.trim().split('=');
      if (k === name) cookieValue = decodeURIComponent(v);
    });
  }
  return cookieValue;
}

export default function UploadClaim() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    claimType: '',
    StayDuration: '',
    cost: '',
    num_diagnoses: '',
    DiagnosisCategory: '',
    num_procedures: '',
    first_procedure: '',
    Gender: '',
    Race: '',
    ClaimDuration: '',
    ClaimDate: '',
    Age: '',
  });

  const handleChange = e => {
    setFormData(f => ({ ...f, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async e => {
    e.preventDefault();

    // build form-encoded body
    const body = new URLSearchParams();
    Object.entries(formData).forEach(([k, v]) => body.append(k, v));

    // grab CSRF token
    const csrftoken = getCookie('csrftoken');

    try {
      const response = await fetch('http://localhost:8000/api/claims/submit/', {
        method: 'POST',
        credentials: 'include',            // send cookie
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRFToken': csrftoken,        // Django CSRF header
        },
        body: body.toString(),
      });

      if (response.ok) {
        console.log('Claim submitted successfully!');
        const username = localStorage.getItem('username');
        navigate(`/profile/${username}`);
  // redirect after success
      } else {
        console.error('Submission failed:', await response.text());
      }
    } catch (err) {
      console.error('Error submitting claim:', err);
    }
  };

  // Function to navigate to the profile page
  const handleBackToProfile = () => {
    navigate(-1);  // Navigate to the profile page of the current user
  };

  return (
    <div className="upload-container">
      <h1>Medical Claim Submission</h1>
      <form onSubmit={handleSubmit} className="upload-form">
        <fieldset className="form-section">
          <legend>Claim Basics</legend>
          <input name="claimType" type="number" placeholder="Claim Type" onChange={handleChange} />
          <input name="ClaimDate" type="date" placeholder="Claim Date" onChange={handleChange} />
          <input name="ClaimDuration" type="number" step="0.01" placeholder="Claim Duration" onChange={handleChange} />
          <input name="cost" type="number" step="0.01" placeholder="Cost" onChange={handleChange} />
          <input name="StayDuration" type="number" step="0.01" placeholder="Stay Duration" onChange={handleChange} />
        </fieldset>

        <fieldset className="form-section">
          <legend>Patient Info</legend>
          <select name="Gender" onChange={handleChange}>
            <option value="">Select Gender</option>
            <option value="0">Other</option>
            <option value="1">Male</option>
            <option value="2">Female</option>
          </select>
          <select name="Race" onChange={handleChange}>
            <option value="">Select Race</option>
            <option value="0">Declined to Answer</option>
            <option value="1">White</option>
            <option value="2">Black or African American</option>
            <option value="3">Other</option>
            <option value="4">Asian / Pacific Islander</option>
            <option value="5">Hispanic</option>
            <option value="6">North American Native</option>
          </select>
          
          <input name="Age" type="number" placeholder="Age" onChange={handleChange} />
        </fieldset>

        <fieldset className="form-section">
          <legend>Diagnosis Info</legend>
          <input name="num_diagnoses" type="number" placeholder="Number of Diagnoses" onChange={handleChange} />
          <input name="DiagnosisCategory" type="number" placeholder="Diagnosis Category (numeric)" onChange={handleChange} />
        </fieldset>

        <fieldset className="form-section">
          <legend>Procedure Info</legend>
          <input name="num_procedures" type="number" placeholder="Number of Procedures" onChange={handleChange} />
          <input name="first_procedure" type="number" placeholder="First Procedure" onChange={handleChange} />
        </fieldset>

      </form>
      <div className="button-row">
        <button className="submit-button" type="submit" onClick={handleSubmit}>Submit Claim</button>
        <button className="back-button" type="button" onClick={handleBackToProfile}>Back to Profile</button>
      </div>
    </div>
  );
}
