import '../styles/ClaimSummary.css';

function ClaimSummary({ claim }) {
  if (!claim) return <p>No claim selected.</p>;

  return (
    <div className="claim-summary">
      <div className="claim-summary-header">Claim Summary</div>

      <div className="summary-block">
        <h3>Basic Info</h3>
        <p><strong>ID:</strong> {claim.id}</p>
        <p><strong>Date:</strong> {claim.claim_date}</p>
        <p><strong>Type:</strong> {claim.claim_type}</p>
        <p><strong>Cost:</strong> ${claim.cost}</p>
        <p><strong>Claim Duration:</strong> {claim.claim_duration}</p>
        <p><strong>Weekend:</strong> {claim.is_weekend ? 'Yes' : 'No'}</p>
      </div>

      <div className="summary-block">
        <h3>Patient Info</h3>
        <p><strong>Gender:</strong> {claim.gender}</p>
        <p><strong>Race:</strong> {claim.race}</p>
        <p><strong>Age:</strong> {claim.age}</p>
      </div>

      <div className="summary-block">
        <h3>Medical Details</h3>
        <p><strong>Stay Duration:</strong> {claim.stay_duration}</p>
        <p><strong>Diagnoses:</strong> {claim.num_diagnoses}</p>
        <p><strong>Diagnosis Category:</strong> {claim.diagnosis_category}</p>
        <p><strong>Procedures:</strong> {claim.num_procedures}</p>
        <p><strong>First Procedure:</strong> {claim.first_procedure}</p>
      </div>

      <div className="summary-block">
        <h3>Predictions</h3>
        <p><strong>NN Prediction:</strong> {claim.nn_prediction} ({claim.nn_label})</p>
        <p><strong>XGB Prediction:</strong> {claim.xgb_prediction} ({claim.xgb_label})</p>
      </div>
    </div>
  );
}

export default ClaimSummary;
