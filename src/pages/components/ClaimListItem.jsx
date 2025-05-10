import { useState } from 'react'

import '../styles/ClaimItem.css'

function ClaimListItem({id, date, isActive, onClick}) {
    return (
        <div className={`claim-item ${isActive ? 'active' : ''}`} onClick={onClick}>
            <h1>Claim ID #{id}</h1>
            <h2>{date}</h2>
        </div>
    )

}
export default ClaimListItem