import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
 
 function UploadClaim() {
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
     isWeekend: '',
     ClaimDuration: '',
     ClaimDate: '',
     Age: '',
     first_diagnosis: '',
   });
 
   const handleChange = (e) => {
     setFormData({ ...formData, [e.target.name]: e.target.value });
   };
 
   const handleSubmit = async (e) => {
     e.preventDefault();
     try {
       const response = await fetch('http://localhost:8000/api/submit-claim/', {  // change URL if needed
         method: 'POST',
         headers: {
           'Content-Type': 'application/json',
         },
         body: JSON.stringify(formData),
       });
 
       if (response.ok) {
         console.log('Claim submitted successfully!');
         navigate('/');  // Redirect after successful submission
       } else {
         console.error('Failed to submit claim');
       }
     } catch (err) {
       console.error('Error submitting claim:', err);
     }
   };
 
   return (
     <div style={{ padding: '2rem' }}>
       <h1>Medical Claim Submission</h1>
       <form onSubmit={handleSubmit}>
         <input name="claimType" type="number" placeholder="Claim Type" onChange={handleChange} /><br />
         <input name="StayDuration" type="number" step="0.01" placeholder="Stay Duration" onChange={handleChange} required /><br />
         <input name="cost" type="number" step="0.01" placeholder="Cost" onChange={handleChange} /><br />
         <input name="num_diagnoses" type="number" placeholder="Number of Diagnoses" onChange={handleChange} /><br />
         <input name="DiagnosisCategory" placeholder="Diagnosis Category" onChange={handleChange} /><br />
         <input name="num_procedures" type="number" placeholder="Number of Procedures" onChange={handleChange} /><br />
         <input name="first_procedure" type="number" placeholder="First Procedure" onChange={handleChange} /><br />
 
         <select name="Gender" onChange={handleChange}>
           <option value="">Select Gender</option>
           <option value="0">Other</option>
           <option value="1">Male</option>
           <option value="2">Female</option>
         </select><br />
 
         <select name="Race" onChange={handleChange}>
           <option value="">Select Race</option>
           <option value="0">Declined to Answer</option>
           <option value="1">White</option>
           <option value="2">Black or African American</option>
           <option value="3">Other</option>
           <option value="4">Asian / Pacific Islander</option>
           <option value="5">Hispanic</option>
           <option value="6">North American Native</option>
         </select><br />
 
         <select name="isWeekend" onChange={handleChange}>
           <option value="">Weekend?</option>
           <option value="0">No</option>
           <option value="1">Yes</option>
         </select><br />
 
         <input name="ClaimDuration" type="number" step="0.01" placeholder="Claim Duration" onChange={handleChange} /><br />
         <input name="ClaimDate" type="date" placeholder="Claim Date" onChange={handleChange} /><br />
         <input name="Age" type="number" placeholder="Age" onChange={handleChange} /><br />
         <input name="first_diagnosis" type="number" placeholder="First Diagnosis" onChange={handleChange} /><br />
 
         <button type="submit">Submit Claim</button>
       </form>
     </div>
   );
 }
 export default UploadClaim;


