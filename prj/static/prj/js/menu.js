(function () { //他のプログラムに影響を与えないようにjsでは即時関数で囲う
    'use strict'; //エラーチェック

    var openMenu = document.getElementById('open_menu');
    var closeMenu = document.getElementById('close_menu');
    var menu = document.getElementById('menu');

    openMenu.addEventListener('click', function () {

        menu.classList.add('shown') //addでshownクラスを追加
    });


    closeMenu.addEventListener('click', function () {

        menu.classList.remove('shown') // removeで取り除く
    });
})();