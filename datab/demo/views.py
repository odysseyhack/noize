from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
                                ListView,DetailView,
                                CreateView,DeleteView,
                                UpdateView)

import datetime
from . import models

class IndexView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self,**kwargs):
        context  = super().get_context_data(**kwargs)
        context['injectme'] = "Basic Injection! "
        return context

class DisasterListView(ListView):
    model = models.Disaster

class DisasterDetailView(DetailView):
    context_object_name = 'disaster_details'
    model = models.Disaster
    template_name = 'demo/disaster_detail.html'


class DisasterCreateView(CreateView):
    fields = ("name")
    model = models.Disaster

class DisasterUpdateView(UpdateView):
    fields = ("name","industry")
    model = models.Disaster

