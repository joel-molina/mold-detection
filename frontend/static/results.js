let breadImage = document.getElementById("bread-image");
let inputFile = document.getElementById("input-file");
let submitButton = document.getElementById("submission");

//display an uploaded file
inputFile.addEventListener("change", () =>
{
    breadImage.src = URL.createObjectURL(inputFile.files[0]);

    //add border.
    breadImage.style.border = "solid";
    breadImage.style.borderRadius = "10px";
    breadImage.style.borderColor = "rgba(156,90,60,255";

    breadImage.alt = "Uploaded image.";
})

//submit button logic
submitButton.addEventListener("click", (e) => 
{
    if(inputFile.files.length > 0)
    {
        processImage(inputFile.files[0]);
    }
    else
    {
        alert("Please upload a file.")
    }
})

//Process image through the backend (through the models)
function processImage(file) 
{
    //add input image to formdata object
    const formData = new FormData();
    formData.append("file", file);

    //send file to backend server
    fetch("/predict", {
        method: "POST",
        body: formData
    })

    //display results
    .then(response => response.json()) //convert json html response to js object

    .then(data => {
        const results = document.querySelector(".result"); //get html .result section
        results.innerHTML = `${data.result}`; //text result from backend

        //dynamic background color.
        if(data.result.includes("No"))
        {
            results.style.backgroundColor = "rgb(66, 182, 245)";
        }
        else
        {
            results.style.backgroundColor = "rgb(245, 141, 66)";
        }
    
        //display image
        const resultImageContainer = document.querySelector(".result-image");
        if (data.image)
        {
            //create image element as base64
            const newImage = document.createElement("img");
            newImage.src = `data:image/jpeg;base64,${data.image}`; //image result from backend
            newImage.alt = "Detection result image";

            //add image
            resultImageContainer.innerHTML = ''; 
            resultImageContainer.appendChild(newImage);
        } 
        else 
        {
            resultImageContainer.innerHTML = '';
        }
    })
}