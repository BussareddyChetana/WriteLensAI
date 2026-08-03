// WriteLens AI Main Client Script

document.addEventListener("DOMContentLoaded", function() {
    // Character and Word Counter for Text Input
    const pastedText = document.getElementById("pasted_text");
    const wordCountBadge = document.getElementById("live_word_count");
    const charCountBadge = document.getElementById("live_char_count");

    if (pastedText && wordCountBadge && charCountBadge) {
        pastedText.addEventListener("input", function() {
            const text = this.value.trim();
            const words = text ? text.split(/\s+/).length : 0;
            const chars = this.value.length;

            wordCountBadge.textContent = `${words} words`;
            charCountBadge.textContent = `${chars} chars`;
        });
    }

    // Input Type Tab Switcher (Text Paste vs File Upload)
    const tabText = document.getElementById("tab-text");
    const tabFile = document.getElementById("tab-file");
    const inputTypeField = document.getElementById("input_type");
    const textGroup = document.getElementById("group-text-input");
    const fileGroup = document.getElementById("group-file-input");

    if (tabText && tabFile && inputTypeField && textGroup && fileGroup) {
        tabText.addEventListener("click", function() {
            tabText.classList.add("active");
            tabFile.classList.remove("active");
            inputTypeField.value = "text";
            textGroup.classList.remove("d-none");
            fileGroup.classList.add("d-none");
        });

        tabFile.addEventListener("click", function() {
            tabFile.classList.add("active");
            tabText.classList.remove("active");
            inputTypeField.value = "file";
            fileGroup.classList.remove("d-none");
            textGroup.classList.add("d-none");
        });
    }

    // Form submit loading spinner
    const analyzeForm = document.getElementById("analyze-form");
    const btnSubmit = document.getElementById("btn-submit-analyze");
    const submitSpinner = document.getElementById("submit-spinner");

    if (analyzeForm && btnSubmit && submitSpinner) {
        analyzeForm.addEventListener("submit", function() {
            btnSubmit.disabled = true;
            submitSpinner.classList.remove("d-none");
        });
    }
});
