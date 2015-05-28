function depoeticize()
{
    $(".depoeticizer-output").html("Depoeticizing...");
    $.ajax({
        type: "GET",
        url: "/apps/dp/depoeticize.cgi",
        dataType: "text",
        data: {
            text: $("#textinput").val(),
            model: $("#corpus").val(),
            errorprob: $("#errorprob").val()
        },
        complete: function (response) {
            if (response.statusText == "OK") {
                var txt = response.responseText;
                $(".depoeticizer-output").html(txt);
            } else {
            }
        }
    });
}
