// Basic client-side validation for the prediction form.
// Prevents negative amounts / obviously invalid values from being submitted.

document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");

    if (form) {
        form.addEventListener("submit", function (event) {
            const amountField = document.getElementById("amount");
            const ageField = document.getElementById("age");
            const hourField = document.getElementById("trans_hour");

            if (amountField && parseFloat(amountField.value) < 0) {
                event.preventDefault();
                alert("Transaction amount cannot be negative.");
                return;
            }

            if (ageField && (ageField.value < 1 || ageField.value > 110)) {
                event.preventDefault();
                alert("Please enter a realistic age.");
                return;
            }

            if (hourField && (hourField.value < 0 || hourField.value > 23)) {
                event.preventDefault();
                alert("Hour must be between 0 and 23.");
            }
        });
    }
});
