var xmlHttp = null;

function refreshStockData(callback)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = processRequest;
    xmlHttp.open("GET", "stocks/collect-stock-data-endpoint/", true); // true for asynchronous
    xmlHttp.send(null);

    $(document).ready(function() {
                $(".toast .toast-body").html("Collecting latest Stock data, this may take a few minutes...");
                $(".toast").toast("show");
                });
}

function processRequest() {
   $(document).ready(function() {
                $(".toast").toast("show");
                });
}

function checkTime(i) {
    if (i < 10) {
        i = "0" + i;
    }
    return i;
}

function startTime() {
    var today = new Date();
    var D = today.getDate();
    var M = today.getMonth();
    var Y = today.getFullYear();
    var h = today.getHours();
    var m = today.getMinutes();
    var s = today.getSeconds();
    // add a zero in front of numbers<10
    m = checkTime(m);
    s = checkTime(s);
    document.getElementById('time').innerHTML = D + "-" + M + "-" +  Y + " " + h + ":" + m + ":" + s;
    t = setTimeout(function () {
        startTime()
    }, 500);
}
startTime();
