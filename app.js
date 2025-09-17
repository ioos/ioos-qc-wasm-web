document.addEventListener("DOMContentLoaded", () => {
  const runQcBtn = document.getElementById("runQcBtn");
  const loader = document.getElementById("loadingIndicator");

  runQcBtn.addEventListener("click", () => {
    if (loader) {
      loader.style.display = "block";
    }
  });
});

const fileInput = document.getElementById('fileInput');
const filenameDisplay = document.getElementById('filename-display');

fileInput.addEventListener('change', function () {
  if (fileInput.files.length > 0) {
    filenameDisplay.textContent = fileInput.files[0].name;
  } else {
    filenameDisplay.textContent = '';
  }
});
