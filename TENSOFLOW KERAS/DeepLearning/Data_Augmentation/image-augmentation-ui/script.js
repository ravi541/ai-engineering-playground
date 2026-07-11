// ================================
// AI Image Augmentation Studio
// script.js
// ================================

// Upload input
const upload = document.getElementById("imageUpload");

// Original image preview
const preview = document.getElementById("preview");

// Gallery
const gallery = document.getElementById("gallery");

// Backend URL
const API_URL = "http://127.0.0.1:8000";


// ====================================
// Preview Uploaded Image
// ====================================

upload.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) return;

    preview.src = URL.createObjectURL(file);

});


// ====================================
// Slider Values
// ====================================

document.querySelectorAll("input[type=range]").forEach(slider => {

    slider.addEventListener("input", function () {

        const valueElement = document.getElementById(this.id + "Value");

        if (!valueElement) return;

        if (this.id === "rotation") {
            valueElement.innerHTML = this.value + "°";
        } else {
            valueElement.innerHTML = this.value + "%";
        }

    });

});


// ====================================
// Generate Images
// ====================================

document.getElementById("generateBtn").addEventListener("click", async function () {

    const file = upload.files[0];

    if (!file) {
        alert("Please upload an image first.");
        return;
    }

    gallery.innerHTML = "<h3>Generating Images...</h3>";

    const formData = new FormData();

    formData.append("file", file);

    try {

        const response = await fetch(API_URL + "/augment", {

            method: "POST",

            body: formData

        });

        if (!response.ok) {

            throw new Error("Backend Error");

        }

        const data = await response.json();

        gallery.innerHTML = "";

        data.generated_images.forEach(imageName => {

            const img = document.createElement("img");

            img.src = API_URL + "/outputs/" + imageName;

            img.width = 180;
            img.height = 180;

            img.style.borderRadius = "10px";
            img.style.margin = "10px";
            img.style.objectFit = "cover";
            img.style.boxShadow = "0px 2px 10px rgba(0,0,0,0.2)";

            gallery.appendChild(img);

        });

        alert("Augmentation Completed Successfully!");

    }

    catch (error) {

        console.error(error);

        gallery.innerHTML = "";

        alert("Cannot connect to FastAPI backend.");

    }

});