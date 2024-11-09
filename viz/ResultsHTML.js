function previous() {
    console.log(sELP_active)
    if (sELP_active) {
        if (counter_sELP > 0) {
            counter_sELP -= 1
            console.log(counter_sELP)
        }
    } else {
        if (counter_wELP > 0) {
            counter_wELP -= 1
            console.log(counter_wELP)
        }
    }
    update_view()
}
function next() {
    console.log(sELP_active)
    if (sELP_active) {
        if (counter_sELP + 1 < n_sELP) {
            counter_sELP += 1
            console.log(counter_sELP)
        }
    } else {
        if (counter_wELP + 1 < n_wELP) {
            counter_wELP+= 1
            console.log(counter_wELP)
        }
    }
    update_view()
}

function numchange() {
    var in_elt = document.getElementById("item_number")
    if (sELP_active) {
        counter_sELP = in_elt.value - 1
    } else {
        counter_wELP= in_elt.value - 1
    }
    update_view()
}

function switch_dataset() {
    var set = document.getElementById("dataset").value
    sELP_active = set === "sELP"
    console.log(set)
    console.log(sELP_active)
    update_view()
}

function switch_model() {
    var model = document.getElementById("model").value
    model_wanted = model
    console.log(model)
    console.log(model_wanted)
    update_view()
}

function imageExists(url) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(true);
        img.onerror = () => resolve(false);
        img.src = url;
    });
}

function name_to_elt(ds, cat, model, name) {
    var img_src = "./assets/"+ds+"/"+model+"/"+cat+"/"+name+"_unified.png"
    if (model == "SIFT" || model =="ExtremeRotations_In_The_Wild"){
        if (ds == "wELP")
        {
            var img_src = "./assets/"+ds+"/"+model+"/"+cat+"/"+name+"_unified.jpg"
        }
    }
    imageExists(img_src).then(exists => {
    if (exists) {
        console.log('Image exists.');
    } 
    });
    console.log(img_src)
    //return `<img src="${img_src}">`;
    //return `<img src="${img_src}" style="width:250%;" onerror="handleError()">`;
    return `<img src="${img_src}" style="width:250%;" onerror="this.onerror=null; this.src=''; document.getElementById('error-message').style.display='block';">
    <p id="error-message" style="display:none; color:red;">The method failed to generate a sufficient number of inliers for estimating the relative rotation.</p>`;
}

function update_view() {
    var c = sELP_active ? counter_sELP : counter_wELP
    var c1 = 1 + c
    var c2 = sELP_active ? n_sELP : n_wELP
    document.getElementById("c2").innerHTML = c2

    var in_elt = document.getElementById("item_number")
    in_elt.value = c1
    in_elt.min = 1
    in_elt.max = sELP_active ? 100 : 100
    

    var model = document.getElementById("model").value
    var ds = sELP_active ? "sELP" : "wELP"
	var cat = document.getElementById("category").value
	
    var name = names[ds][cat][c]
    //var name = "1"


    var elt_img = document.getElementById("results_img")
    elt_img.onerror = function() {
        elt_img.innerHTML = '<p>Image could not be loaded. Please try again later.</p>';
    };

    elt_img.innerHTML = name_to_elt(ds, cat, model, name)

    var elt = document.getElementById("results")
    elt.innerHTML = ''

    var error = errors[ds][cat][model][name]


    elt.innerHTML += '<b>Geodesic error: </b>' + error + '<br><br>'




}
