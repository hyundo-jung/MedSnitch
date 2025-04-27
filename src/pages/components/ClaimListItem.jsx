import { useState } from 'react'

import '../styles/ClaimItem.css'

function ClaimListItem({name, date, claimID, OnClick, isActive}) {
    return (
        <div className={`claim-item ${isActive ? 'active' : ''}`} onClick={OnClick}>
            <h1>{name}</h1>
            <h2>{date}, {claimID}</h2>
        </div>
    )

}
export default ClaimListItem