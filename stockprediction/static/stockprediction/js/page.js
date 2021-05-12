var xmlHttp = null;

// Function to send HTTP request to endpoint to start stock data collection task
function refreshStockData(callback)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = processRequest;
    xmlHttp.open("GET", "collect-stock-data-endpoint", true); // true for asynchronous
    xmlHttp.send(null);

    $(document).ready(function() {
                $(".toast .toast-body").html("Collecting latest Stock data, this may take a few minutes...");
                $(".toast").toast("show");
                });
}

// Function thats called by the refreshStockData() to display toast informing
// the user the collection task has started
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

// Function to start timer in the navbar to displauy current time to user
function startTime() {
    var today = new Date();
    var D = today.getDate();
    var M = today.getMonth();
    var Y = today.getFullYear();
    var h = today.getHours();
    var m = today.getMinutes();
    var s = today.getSeconds();

    // add a zero in front of numbers<10 for consistency
    m = checkTime(m);
    s = checkTime(s);
    document.getElementById('time').innerHTML = D + "-" + M + "-" +  Y + " " + h + ":" + m + ":" + s;
    t = setTimeout(function () {
        startTime()
    }, 500);
}
