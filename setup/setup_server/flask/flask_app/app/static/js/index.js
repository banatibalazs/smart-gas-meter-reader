function check_start_end_values() {
    x = document.getElementById("startDate")
    y = document.getElementById("endDate")

    st = new Date(x.value)
    en = new Date(y.value)

    if (st > en){
    y.value = x.value
    }
}

function toggleDiagramTable(x){
    table = document.getElementById("demo");
    pagination = document.getElementById("pagination");
    graph = document.getElementById("graph");

    if (table.style.display != "none"){
        table.style.display = "none";
        graph.style.display = "block";
        pagination.style.display = "none"
        x.innerHTML = '<i class="fa-solid fa-chart-line"></i> Diagram';
    }
    else {
        table.style.display = "block";
        pagination.style.display = "flex"
        x.innerHTML = '<i class="fa-solid fa-list"></i> Táblázat';
        graph.style.display = "none";
    }
    console.log(x);
}

function errorBtnClick(){
    var errorDiv = document.getElementById("errorDiv");

    if (errorDiv.style.display == 'none') {
    errorDiv.style.display = 'block';
    } else {
    errorDiv.style.display = 'none';
    }

    var errorOption1 = document.getElementById("errorOption1");
    var errorOption2 = document.getElementById("errorOption2");
    var errorInputNumber = document.getElementById("errorInputNumber");
    var errorOldValue = document.getElementById("errorOldValue");
    var errorImg = document.getElementById("errorImg");

    var last_date = document.getElementById("last_date").innerHTML;
    var first_date = document.getElementById("first_date").innerHTML;
    var first_value = document.getElementById("first_v").innerHTML;
    var first_image = document.getElementById("first_image").src;

    errorOption2.innerHTML = last_date;
    errorOption1.innerHTML = first_date;
    errorInputNumber.value = first_value;
    errorOldValue.innerHTML = first_value;
    errorImg.src = first_image;

//    document.getElementById("first_time").innerHTML;
    document.getElementById("first_v").innerHTML;
    document.getElementById("first_image").src;

//    document.getElementById("last_time").innerHTML;
    document.getElementById("last_v").innerHTML;
    document.getElementById("last_image").src;


}

function errorSelect(that){
//    console.log(that.value)
    var errorInputNumber = document.getElementById("errorInputNumber");
    var errorOldValue = document.getElementById("errorOldValue");
    var errorImg = document.getElementById("errorImg");

    if (that.value == 1){


        var first_value = document.getElementById("first_v").innerHTML;
        var first_image = document.getElementById("first_image").src;

        errorInputNumber.value = first_value;
        errorOldValue.innerHTML = first_value;
        errorImg.src = first_image;

    } else {
        var errorInputNumber = document.getElementById("errorInputNumber");
        var errorImg = document.getElementById("errorImg");

        var last_value = document.getElementById("last_v").innerHTML;
        var last_image = document.getElementById("last_image").src;

        errorInputNumber.value = last_value;
        errorImg.src = last_image;
        errorOldValue.innerHTML = last_value;
    }
}

function changeImgValue(){
    var value = document.getElementById("errorInputNumber").value;
    var img = document.getElementById("errorImg").src;

//    console.log(value);
    value = Math.round(value)
    var errorInputNumber = document.getElementById("errorInputNumber").value;
    var errorOldValue = parseInt(document.getElementById("errorOldValue").innerHTML);

//    console.log(value);
    if (value > 13700 && value < 99999 && errorInputNumber != errorOldValue){


    slices = img.split("/")
    slice = slices.pop()

//    img = slices[-1]
//    img = img.slice(0,-4)

//    console.log(img);
//    console.log(slice);

    var url = '/changeRequest?img=' + slice + '&value=' + value;
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", url, false ); // false for synchronous request
    xmlHttp.send( null );

    if ( xmlHttp.status == 200){
    var start_date = document.getElementById("first_date").innerHTML;
    var end_date = document.getElementById("last_date").innerHTML;

    var url = '/filter?startDate=' + start_date + '&endDate=' + end_date;
    window.location.href = url;
    }
    } else {
        console.log(value);
    }

}


function trClick(that){

    var errorDiv = document.getElementById("errorDiv");
    errorDiv.style.display = 'none';


//    console.log(document.getElementsByClassName('picked').length);
//    console.log(document.getElementsByClassName('picked')[0]);
    if (document.getElementsByClassName('picked').length > 0){
    document.getElementsByClassName('picked')[0].classList.remove('picked');
    }
    that.classList.add("picked");
//    console.log(document.getElementsByClassName('picked').length);


//    console.log(that.children[1]);
    var year = that.children[0].children[0].innerHTML;
    var month = that.children[1].children[0].innerHTML;

    const thirtyOneMonths = ["01", "03", "05", "07", "08","10","12"];
    const thirtyMonths = ["04", "06", "09", "11"];
    if ( thirtyOneMonths.includes(month)){
        day = '31';
    }
    else if (thirtyMonths.includes(month)){
        day = '30';
    } else {
        day = '28';
    }
    var end_date = year + '-' + month + '-' + day;
    var start_date = year + '-' + month + '-01';
//    console.log(start_date);
//    console.log(end_date);

    var url = '/dateRequest?startDate=' + start_date + '&endDate=' + end_date;
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", url, false ); // false for synchronous request
    xmlHttp.send( null );

//    console.log(JSON.parse(xmlHttp.response));

    var resp = JSON.parse(xmlHttp.responseText);

    var first_image = resp['first_image'];
    var first_v = resp['first_v'];
    var first_date = resp['first_date'];
    var first_time = resp['first_time'];
    var last_image = resp['last_image'];
    var last_v = resp['last_v='];
    var last_date = resp['last_date'];
    var last_time = resp['last_time'];

    var time_delta = resp['time_delta'];
    var value_delta = resp['value_delta'];
    var average_value = resp['average_value'];

//    console.log(first_image)

    document.getElementById("first_date").innerHTML = first_date;
    document.getElementById("first_time").innerHTML = first_time;
    document.getElementById("first_v").innerHTML = first_v;
    document.getElementById("first_image").src = first_image;

    document.getElementById("last_date").innerHTML = last_date;
    document.getElementById("last_time").innerHTML = last_time;
    document.getElementById("last_v").innerHTML= last_v;
    document.getElementById("last_image").src = last_image;

    document.getElementById("time_delta").innerHTML = time_delta;
    document.getElementById("value_delta").innerHTML = value_delta;
    document.getElementById("average_value").innerHTML = average_value;
}


//function trClick(that){
//    console.log(that.children[1]);
//    var year = that.children[0].children[0].innerHTML;
//    var month = that.children[1].children[0].innerHTML;
//
//    const thirtyOneMonths = ["01", "03", "05", "07", "08","10","12"];
//    const thirtyMonths = ["04", "06", "09", "11"];
//    if ( thirtyOneMonths.includes(month)){
//        day = '31';
//    }
//    else if (thirtyMonths.includes(month)){
//        day = '30';
//    } else {
//        day = '28';
//    }
//    var end_date = year + '-' + month + '-' + day;
//    var start_date = year + '-' + month + '-01';
//    console.log(start_date);
//    console.log(end_date);
//
//    var url = '/filter?startDate=' + start_date + '&endDate=' + end_date;
//    window.location.href = url;
//
//}