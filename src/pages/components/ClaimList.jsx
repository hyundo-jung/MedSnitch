import { useState } from 'react'
import ClaimListItem from './ClaimListItem'

import '../styles/ClaimList.css'

function ClaimList({ data, handleClaimClick, activeClaim }) {
   return (
    <>
        <p className="claimlist-header">Past Claim Uploads</p>
        {data.map((claim) => (
            <ClaimListItem 
                key={claim.id}
                id={claim.id}
                date={claim.claim_date} 
                isActive={activeClaim === claim.id}
                onClick={()=>handleClaimClick(claim.id)}
            />))}
        </>
   )

}
export default ClaimList