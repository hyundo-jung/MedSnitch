import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useNavigate } from 'react-router-dom'
import ClaimList from './components/ClaimList'
import ClaimSummary from './components/ClaimSummary'
import './styles/ProfilePage.css'

function ProfilePage() {
    const { username } = useParams();
    const navigate = useNavigate(); 
    
    const [activeClaim, setActiveClaim] = useState(-1) //claim id of active (clicked claim)
    const [claims, setClaims] = useState([]);

    // const fakeData = [{name: "claim1", date: '1/2/2000', claimID: 1}, 
    // {name: "claim2", date: '4/7/2000', claimID: 2}, 
    // {name: "claim3", date: '12/3/2000', claimID: 3}]

    useEffect(() => {
        fetch('http://localhost:8000/api/claims/users/', {
          method: 'GET',
          credentials: 'include',  // ESSENTIAL
        })
          .then(response => {
            if (!response.ok) throw new Error('Failed to fetch claims');
            return response.json();
          })
          .then(data => {
            console.log('Fetched claims:', data.claims); // ← log this
            setClaims(data.claims); // Assuming backend sends { claims: [...] }
          })
          .catch(error => {
            console.error('Error fetching user claims:', error);
          });
      }, []);

    const handleClaimClick = (id) => setActiveClaim(id); 
    const handleUploadClick = () => {
        navigate('/upload-claim');
    };

    const handleLogout = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/accounts/logout/', {
          method: 'POST',
          credentials: 'include',
        });
    
        if (response.ok) {
          localStorage.removeItem('username'); // Clear stored username if you used it
          navigate('/'); // Go back to homepage or login
        } else {
          console.error('Logout failed');
        }
      } catch (err) {
        console.error('Error logging out:', err);
      }
    };
    

    return (
        <div className="profile-page">
            <div className="profile-header">
                <h1 className="profile-heading">Welcome, {username}</h1>
                <button className="upload-button" onClick={handleUploadClick}>Upload New Claim</button>
                <button className="logout-button" onClick={handleLogout}>Log Out</button>
            </div>
            <div className="profile-claims">
                <div className="profile-claimlist">
                    <ClaimList data={claims} handleClaimClick={handleClaimClick} activeClaim={activeClaim}/>
                </div>
                <div className="profile-claim-summary">
                    {(activeClaim !== -1) ? <ClaimSummary claim={claims.find(c => c.id === activeClaim)}/> : <p className="placeholder"> click on a claim to view stats</p>}
                </div>
            </div>
        </div>
        
    )
}
export default ProfilePage