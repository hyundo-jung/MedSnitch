import { useState } from 'react'
import ClaimListItem from './ClaimListItem'

import '../styles/ClaimList.css'

function ClaimList({ data, handleClaimClick, activeClaim }) {
   return (
    <>
        <p className="claimlist-header">Past Claim Uploads</p>
        {data.map((claim) => (
            <ClaimListItem 
                key={claim.claimID}
                name={claim.name} 
                date={claim.date} 
                claimID={claim.claimID} 
                isActive={activeClaim === claim.claimID}
                OnClick={()=>handleClaimClick(claim.claimID)}
            />))}
        </>
   )

}
export default ClaimList