import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import ClaimList from './components/ClaimList'
import ClaimSummary from './components/ClaimSummary'
import './styles/ProfilePage.css'

function ProfilePage() {
    const fakeData = [{name: "claim1", date: '1/2/2000', claimID: 1}, 
    {name: "claim2", date: '4/7/2000', claimID: 2}, 
    {name: "claim3", date: '12/3/2000', claimID: 3}]
    const navigate = useNavigate(); 
    const [activeClaim, setActiveClaim] = useState(-1) //claim id of active (clicked claim)

    const handleClaimClick = (claimID) => setActiveClaim(claimID); 
    const handleUploadClick = () => {
        window.location.href = 'http://localhost:8000/claims/submit/';
    };

    return (
        <div className="profile-page">
            <div className="profile-header">
                <h1 className="profile-heading">Welcome, Alex</h1>
                <button className="upload-button" onClick={handleUploadClick}>Upload New Claim</button>
            </div>
            <div className="profile-claims">
                <div className="profile-claimlist">
                    <ClaimList data={fakeData} handleClaimClick={handleClaimClick} activeClaim={activeClaim}/>
                </div>
                <div className="profile-claim-summary">
                    {(activeClaim !== -1) ? <ClaimSummary claimID={activeClaim}/> : <p className="placeholder"> click on a claim to view stats</p>}
                </div>
            </div>
        </div>
        
    )
}
export default ProfilePage