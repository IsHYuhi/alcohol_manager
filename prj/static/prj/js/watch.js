(function () {
    'use strict';

    var timer = document.getElementById('timer');
    var start = document.getElementById('start');
    var stop = document.getElementById('stop');
    var reset = document.getElementById('reset');

    var startTime;
    var elapsedTime = 0;
    var timerId;
    var timeToAdd = 0;
    var isRunning = false;

    function updateTimerText() {
        // 135200 -> 02:15.200
        // m = 135200 / 60000 の商
        // s = 135200 % 60000 / 1000
        // ms = 135200 % 1000
        var m = Math.floor(elapsedTime / 60000);
        var s = Math.floor(elapsedTime % 60000 / 1000);
        var ms = elapsedTime % 1000;

        //  3 -> '03'
        // 12 -> '12'

        //  3 -> '0' +  3 -> '03'
        // 12 -> '0' + 12 -> '012'

        m = ('0' + m).slice(-2);
        s = ('0' + s).slice(-2);
        ms = ('00' + ms).slice(-3);

        timer.textContent = m + ':' + s;
    }

    function countUp() {
        timerId = setTimeout(function () {
            // elapsedTime = Date.now() - startTime;
            elapsedTime = Date.now() - startTime + timeToAdd;
            // console.log(elapsedTime);
            updateTimerText();
            countUp();
        }, 10);
    }

    // start.className = 'btn';
    // stop.className = 'btn inactive';
    // reset.className = 'btn inactive';

    function updateButtonState(startButtonState, stopButtonState, resetButtonState) {
        start.className = startButtonState ? 'btn' : 'btn inactive';
        stop.className = stopButtonState ? 'btn' : 'btn inactive';
        reset.className = resetButtonState ? 'btn' : 'btn inactive';
    }

    updateButtonState(true, false, false);

    start.addEventListener('click', function () {
        if (isRunning === true) {
            return;
        }
        isRunning = true;
        // start: false, stop: true, reset: false
        updateButtonState(false, true, false);
        startTime = Date.now();
        countUp();
    });

    stop.addEventListener('click', function () {
        if (isRunning === false) {
            return;
        }
        isRunning = false;
        // start: true, stop: false, reset: true
        updateButtonState(true, false, true);
        clearTimeout(timerId);
        timeToAdd += Date.now() - startTime;
    });

    reset.addEventListener('click', function () {
        if (isRunning === true) {
            return;
        }
        // start: true, stop: false, reset: false
        updateButtonState(true, false, false);
        elapsedTime = 0;
        timeToAdd = 0;
        updateTimerText();
    });
})();