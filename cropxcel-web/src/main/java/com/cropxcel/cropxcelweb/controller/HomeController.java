package com.cropxcel.cropxcelweb.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    @GetMapping({ "/", "/home" })
    public String home(Model model) {
        return "index";
    }

    @GetMapping("/prediction")
    public String showPredictionPage() {
        return "prediction"; // return the name of the prediction page's HTML file
    }
}
