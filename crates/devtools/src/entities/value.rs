//! Drawing of reflected values, such as a component's fields. An edit is
//! recorded as the new value and the path to it, which [`element_mut`]
//! follows.

use std::any::TypeId;
use std::borrow::Cow;
use std::fmt::{self, Write as _};
use std::time::Duration;

use bevy::math::EulerRot;
use bevy::prelude::*;
use bevy::reflect::enums::{DynamicEnum, DynamicVariant, Enum, VariantInfo, VariantType};
use bevy::reflect::std_traits::ReflectDefault;
use bevy::reflect::structs::DynamicStruct;
use bevy::reflect::tuple::DynamicTuple;
use bevy::reflect::{PartialReflect, ReflectMut, ReflectRef, TypeRegistry};
use pumicite_egui::egui::{self, emath::Numeric};

use super::{Action, INDENT, Syntax, WorldInfo, collapsible_row, entity_link};

/// The most elements of a list, map or set drawn; the rest are counted.
pub(super) const MAX_ELEMENTS: usize = 100;

/// The value at `path` within `value`. Each index in `path` picks a field of
/// a struct, tuple or enum variant, or an element of a list or array, as
/// [`ValueUi`] records them.
pub(super) fn element_mut<'a>(
    value: &'a mut dyn PartialReflect,
    path: &[usize],
) -> Option<&'a mut dyn PartialReflect> {
    let Some((&index, rest)) = path.split_first() else {
        return Some(value);
    };
    let element = match value.reflect_mut() {
        ReflectMut::Struct(value) => value.field_at_mut(index),
        ReflectMut::TupleStruct(value) => value.field_mut(index),
        ReflectMut::Tuple(value) => value.field_mut(index),
        ReflectMut::List(value) => value.get_mut(index),
        ReflectMut::Array(value) => value.get_mut(index),
        ReflectMut::Enum(value) => value.field_at_mut(index),
        _ => None,
    }?;
    element_mut(element, rest)
}

/// `value` as its `Debug` implementation formats it, on one line and cut at
/// `max` bytes. Formatting stops at the cut, so that previewing a large value
/// costs no more than a small one.
pub(super) fn preview(value: &dyn PartialReflect, max: usize) -> String {
    struct Bounded {
        text: String,
        max: usize,
    }
    impl fmt::Write for Bounded {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            for c in s.chars() {
                if self.text.len() >= self.max {
                    return Err(fmt::Error);
                }
                self.text.push(if c.is_whitespace() { ' ' } else { c });
            }
            Ok(())
        }
    }
    let mut bounded = Bounded {
        text: String::new(),
        max,
    };
    if write!(bounded, "{value:?}").is_err() {
        bounded.text.push('…');
    }
    bounded.text
}

/// Draws a component's value and records the user's edits to it.
pub(super) struct ValueUi<'a, 'w> {
    info: &'a WorldInfo<'w>,
    registry: &'a TypeRegistry,
    /// Whether the value drawn can be edited: `false` within an immutable
    /// component, and within maps and sets, whose elements `element_mut`
    /// can't reach.
    editable: bool,
    /// Indices from the component down to the value being drawn, as
    /// `element_mut` follows them.
    path: Vec<usize>,
    /// Keys the open state of nested values. Derived from the component's
    /// type, so that the same fields are open whichever entity shows.
    id: egui::Id,
    /// New values, each with its path.
    edits: Vec<(Vec<usize>, Box<dyn PartialReflect>)>,
    /// Where entity links record their clicks.
    actions: &'a mut Vec<Action>,
}

impl<'a, 'w> ValueUi<'a, 'w> {
    pub(super) fn new(
        info: &'a WorldInfo<'w>,
        registry: &'a TypeRegistry,
        editable: bool,
        id: egui::Id,
        actions: &'a mut Vec<Action>,
    ) -> Self {
        Self {
            info,
            registry,
            editable,
            path: Vec::new(),
            id,
            edits: Vec::new(),
            actions,
        }
    }

    /// The edits recorded, each a new value and the path to it.
    pub(super) fn into_edits(self) -> Vec<(Vec<usize>, Box<dyn PartialReflect>)> {
        self.edits
    }

    /// Draws a component's value: on one row if it fits, or else a row per
    /// field or element, after the variant for an enum.
    pub(super) fn show(&mut self, ui: &mut egui::Ui, value: &dyn PartialReflect) {
        if self.is_inline(value) {
            ui.horizontal_wrapped(|ui| {
                ui.add_space(INDENT);
                self.inline(ui, value);
            });
            return;
        }
        if let ReflectRef::Enum(value) = value.reflect_ref() {
            ui.horizontal(|ui| {
                ui.add_space(INDENT);
                self.variant(ui, value);
            });
        }
        self.elements(ui, value);
    }

    fn edit(&mut self, value: Box<dyn PartialReflect>) {
        self.edits.push((self.path.clone(), value));
    }

    /// Whether `value` fits on one row: a single value, a vector of numbers,
    /// an enum variant with at most one such field, or an empty collection.
    fn is_inline(&self, value: &dyn PartialReflect) -> bool {
        match value.reflect_ref() {
            ReflectRef::Struct(value) => {
                value.field_len() == 0 || is_vector(value.iter_fields().map(|(_, field)| field))
            }
            ReflectRef::TupleStruct(value) => {
                value.field_len() == 0
                    || is_vector(value.iter_fields())
                    || (value.field_len() == 1
                        && value.field(0).is_some_and(|field| self.is_inline(field)))
            }
            ReflectRef::Tuple(value) => value.field_len() == 0 || is_vector(value.iter_fields()),
            ReflectRef::Enum(value) => {
                value.field_len() == 0
                    || (value.variant_type() == VariantType::Tuple
                        && value.field_len() == 1
                        && value.field_at(0).is_some_and(|field| self.is_inline(field)))
            }
            ReflectRef::List(value) => value.is_empty(),
            ReflectRef::Array(value) => value.is_empty(),
            ReflectRef::Map(value) => value.is_empty(),
            ReflectRef::Set(value) => value.is_empty(),
            _ => true,
        }
    }

    /// One row for a field: its name, then its value on the same row if
    /// [`Self::is_inline`], or else below it, collapsed.
    fn field(&mut self, ui: &mut egui::Ui, name: &str, value: &dyn PartialReflect) {
        if self.is_inline(value) {
            ui.horizontal_wrapped(|ui| {
                // Lines the name up with those of collapsible fields, which
                // start after a disclosure triangle.
                ui.add_space(INDENT);
                field_name(ui, name);
                self.inline(ui, value);
            });
        } else {
            self.nested(ui, name, value);
        }
    }

    /// A collapsible row for a value that doesn't fit on one: closed, it
    /// shows a preview of the value; open, its fields or elements below.
    fn nested(&mut self, ui: &mut egui::Ui, name: &str, value: &dyn PartialReflect) {
        let id = self.id.with(&self.path);
        let (_, open) = collapsible_row(ui, id, false, egui::Color32::TRANSPARENT, |ui| {
            field_name(ui, name);
            match value.reflect_ref() {
                // The variant can be picked without opening the row.
                ReflectRef::Enum(value) => self.variant(ui, value),
                reflected => {
                    let count = match reflected {
                        ReflectRef::List(value) => Some(value.len()),
                        ReflectRef::Array(value) => Some(value.len()),
                        ReflectRef::Map(value) => Some(value.len()),
                        ReflectRef::Set(value) => Some(value.len()),
                        _ => None,
                    };
                    if let Some(count) = count {
                        ui.weak(format!("({count})"));
                    }
                    ui.add(
                        egui::Label::new(egui::RichText::new(preview(value, 200)).weak())
                            .truncate(),
                    );
                }
            }
        });
        if open {
            ui.indent(id, |ui| self.elements(ui, value));
        }
    }

    /// A row per field or element of `value`.
    fn elements(&mut self, ui: &mut egui::Ui, value: &dyn PartialReflect) {
        let mut more = 0;
        match value.reflect_ref() {
            ReflectRef::Struct(value) => {
                for (index, (name, element)) in value.iter_fields().enumerate() {
                    self.element(ui, index, name, element);
                }
            }
            ReflectRef::TupleStruct(value) => {
                for (index, element) in value.iter_fields().enumerate() {
                    self.element(ui, index, &index.to_string(), element);
                }
            }
            ReflectRef::Tuple(value) => {
                for (index, element) in value.iter_fields().enumerate() {
                    self.element(ui, index, &index.to_string(), element);
                }
            }
            ReflectRef::Enum(value) => {
                for (index, element) in value.iter_fields().enumerate() {
                    let name = match element.name() {
                        Some(name) => Cow::Borrowed(name),
                        None => Cow::Owned(index.to_string()),
                    };
                    self.element(ui, index, &name, element.value());
                }
            }
            ReflectRef::List(value) => {
                for (index, element) in value.iter().take(MAX_ELEMENTS).enumerate() {
                    self.element(ui, index, &index.to_string(), element);
                }
                more = value.len().saturating_sub(MAX_ELEMENTS);
            }
            ReflectRef::Array(value) => {
                for (index, element) in value.iter().take(MAX_ELEMENTS).enumerate() {
                    self.element(ui, index, &index.to_string(), element);
                }
                more = value.len().saturating_sub(MAX_ELEMENTS);
            }
            ReflectRef::Map(value) => {
                let editable = std::mem::replace(&mut self.editable, false);
                for (index, (key, element)) in value.iter().take(MAX_ELEMENTS).enumerate() {
                    self.element(ui, index, &preview(key, 40), element);
                }
                self.editable = editable;
                more = value.len().saturating_sub(MAX_ELEMENTS);
            }
            ReflectRef::Set(value) => {
                let editable = std::mem::replace(&mut self.editable, false);
                for (index, element) in value.iter().take(MAX_ELEMENTS).enumerate() {
                    self.element(ui, index, &index.to_string(), element);
                }
                self.editable = editable;
                more = value.len().saturating_sub(MAX_ELEMENTS);
            }
            _ => {
                ui.horizontal_wrapped(|ui| self.inline(ui, value));
            }
        }
        if more > 0 {
            ui.horizontal(|ui| {
                ui.add_space(INDENT);
                ui.weak(format!("… {more} more"));
            });
        }
    }

    /// The row of the field or element at `index` of the value drawn.
    fn element(&mut self, ui: &mut egui::Ui, index: usize, name: &str, value: &dyn PartialReflect) {
        self.path.push(index);
        self.field(ui, name, value);
        self.path.pop();
    }

    /// `value` on the current row, after its field's name.
    fn inline(&mut self, ui: &mut egui::Ui, value: &dyn PartialReflect) {
        if self.leaf(ui, value) {
            return;
        }
        match value.reflect_ref() {
            ReflectRef::Struct(fields) if fields.field_len() > 0 => {
                for (index, (name, field)) in fields.iter_fields().enumerate() {
                    ui.weak(name);
                    self.path.push(index);
                    self.inline(ui, field);
                    self.path.pop();
                }
            }
            ReflectRef::TupleStruct(fields) if fields.field_len() == 1 => {
                // A newtype shows as the value it wraps.
                self.path.push(0);
                self.inline(ui, fields.field(0).expect("one field"));
                self.path.pop();
            }
            ReflectRef::TupleStruct(fields) if fields.field_len() > 0 => {
                for (index, field) in fields.iter_fields().enumerate() {
                    ui.weak(index.to_string());
                    self.path.push(index);
                    self.inline(ui, field);
                    self.path.pop();
                }
            }
            ReflectRef::Tuple(fields) if fields.field_len() > 0 => {
                for (index, field) in fields.iter_fields().enumerate() {
                    ui.weak(index.to_string());
                    self.path.push(index);
                    self.inline(ui, field);
                    self.path.pop();
                }
            }
            ReflectRef::Enum(value) => {
                self.variant(ui, value);
                if let Some(field) = value.field_at(0) {
                    self.path.push(0);
                    self.inline(ui, field);
                    self.path.pop();
                }
            }
            ReflectRef::List(_) | ReflectRef::Array(_) => {
                ui.weak("[]");
            }
            ReflectRef::Map(_) | ReflectRef::Set(_) => {
                ui.weak("{}");
            }
            ReflectRef::Struct(_) | ReflectRef::TupleStruct(_) | ReflectRef::Tuple(_) => {
                // A unit struct; its type is all there is to it.
                ui.weak(value.reflect_short_type_path());
            }
            _ => {
                // An opaque type without a widget of its own.
                ui.label(egui::RichText::new(preview(value, 200)).monospace().weak());
            }
        }
    }

    /// Draws `value` if it's of a type with a widget of its own, and returns
    /// whether it was.
    fn leaf(&mut self, ui: &mut egui::Ui, value: &dyn PartialReflect) -> bool {
        if self.number::<f32>(ui, value)
            || self.number::<f64>(ui, value)
            || self.number::<i8>(ui, value)
            || self.number::<i16>(ui, value)
            || self.number::<i32>(ui, value)
            || self.number::<i64>(ui, value)
            || self.number::<isize>(ui, value)
            || self.number::<u8>(ui, value)
            || self.number::<u16>(ui, value)
            || self.number::<u32>(ui, value)
            || self.number::<u64>(ui, value)
            || self.number::<usize>(ui, value)
        {
            return true;
        }
        let visuals = ui.visuals();
        let number_color = Syntax::Number.color(visuals);
        let string_color = Syntax::Value.color(visuals);
        if let Some(&value) = value.try_downcast_ref::<bool>() {
            if self.editable {
                let mut edited = value;
                if ui.checkbox(&mut edited, "").changed() {
                    self.edit(Box::new(edited));
                }
            } else {
                ui.label(egui::RichText::new(value.to_string()).color(number_color));
            }
        } else if let Some(value) = value.try_downcast_ref::<String>() {
            if self.editable {
                let mut edited = value.clone();
                let field = egui::TextEdit::singleline(&mut edited).desired_width(f32::INFINITY);
                if ui.add(field).changed() {
                    self.edit(Box::new(edited));
                }
            } else {
                ui.label(egui::RichText::new(format!("{value:?}")).color(string_color));
            }
        } else if let Some(value) = value
            .try_downcast_ref::<Cow<'static, str>>()
            .map(|value| &**value)
            .or_else(|| value.try_downcast_ref::<&'static str>().copied())
        {
            ui.label(egui::RichText::new(format!("{value:?}")).color(string_color));
        } else if let Some(value) = value.try_downcast_ref::<char>() {
            ui.label(egui::RichText::new(format!("{value:?}")).color(string_color));
        } else if let Some(value) = value.try_downcast_ref::<i128>() {
            ui.label(egui::RichText::new(value.to_string()).color(number_color));
        } else if let Some(value) = value.try_downcast_ref::<u128>() {
            ui.label(egui::RichText::new(value.to_string()).color(number_color));
        } else if let Some(&entity) = value.try_downcast_ref::<Entity>() {
            entity_link(ui, self.info, entity, self.actions);
        } else if let Some(&duration) = value.try_downcast_ref::<Duration>() {
            if self.editable {
                let mut seconds = duration.as_secs_f64();
                let field = egui::DragValue::new(&mut seconds)
                    .speed(0.01)
                    .range(0.0..=f64::MAX)
                    .suffix(" s");
                // Seconds past what a `Duration` holds are ignored.
                if ui.add(field).changed()
                    && let Ok(duration) = Duration::try_from_secs_f64(seconds)
                {
                    self.edit(Box::new(duration));
                }
            } else {
                ui.label(egui::RichText::new(format!("{duration:?}")).color(number_color));
            }
        } else if let Some(&rotation) = value.try_downcast_ref::<Quat>() {
            self.rotation(ui, rotation);
        } else {
            return false;
        }
        true
    }

    /// Draws `value` if it's a `T`, and returns whether it was.
    fn number<T: Numeric + PartialReflect + fmt::Display>(
        &mut self,
        ui: &mut egui::Ui,
        value: &dyn PartialReflect,
    ) -> bool {
        let Some(&value) = value.try_downcast_ref::<T>() else {
            return false;
        };
        if !self.editable {
            let color = Syntax::Number.color(ui.visuals());
            ui.label(egui::RichText::new(value.to_string()).color(color));
            return true;
        }
        // A pixel of dragging moves a float by a hundredth of its magnitude,
        // so that small and large values both edit smoothly.
        let speed = if T::INTEGRAL {
            0.25
        } else {
            (value.to_f64().abs() * 0.01).max(0.001)
        };
        let mut edited = value;
        if ui
            .add(egui::DragValue::new(&mut edited).speed(speed))
            .changed()
        {
            self.edit(Box::new(edited));
        }
        true
    }

    /// A rotation as Euler angles in degrees, which are easier to read and
    /// edit than a quaternion's components.
    fn rotation(&mut self, ui: &mut egui::Ui, rotation: Quat) {
        let (x, y, z) = rotation.to_euler(EulerRot::XYZ);
        let mut angles = [x, y, z].map(f32::to_degrees);
        let mut changed = false;
        for (axis, angle) in ["x", "y", "z"].into_iter().zip(&mut angles) {
            ui.weak(axis);
            if self.editable {
                let field = egui::DragValue::new(angle).speed(0.5).suffix("°");
                changed |= ui
                    .add(field)
                    .on_hover_text("Euler angle, XYZ order")
                    .changed();
            } else {
                let color = Syntax::Number.color(ui.visuals());
                ui.label(egui::RichText::new(format!("{angle:.1}°")).color(color));
            }
        }
        if changed {
            let [x, y, z] = angles.map(f32::to_radians);
            self.edit(Box::new(Quat::from_euler(EulerRot::XYZ, x, y, z)));
        }
    }

    /// The enum's variant, picked from a combo box when editable. A variant
    /// picked whose fields have no reflected `Default` can't be built, and is
    /// disabled.
    fn variant(&mut self, ui: &mut egui::Ui, value: &dyn Enum) {
        let current = value.variant_name();
        let info = value.get_represented_enum_info().filter(|_| self.editable);
        let Some(info) = info else {
            let color = Syntax::Value.color(ui.visuals());
            ui.label(egui::RichText::new(current).color(color));
            return;
        };
        let mut picked = None;
        egui::ComboBox::from_id_salt(self.id.with(&self.path).with("variant"))
            .selected_text(current)
            .show_ui(ui, |ui| {
                for variant in info.iter() {
                    let name = variant.name();
                    if name == current {
                        ui.add(egui::Button::selectable(true, name));
                        continue;
                    }
                    match default_variant(variant, self.registry) {
                        Some(value) => {
                            if ui.add(egui::Button::selectable(false, name)).clicked() {
                                picked = Some(value);
                            }
                        }
                        None => {
                            ui.add_enabled(false, egui::Button::selectable(false, name))
                                .on_disabled_hover_text(
                                    "Some of the variant's fields don't reflect `Default`.",
                                );
                        }
                    }
                }
            });
        if let Some(value) = picked {
            self.edit(Box::new(value));
        }
    }
}

/// A field's name, before its value.
pub(super) fn field_name(ui: &mut egui::Ui, name: &str) {
    let color = Syntax::Attribute.color(ui.visuals());
    ui.label(egui::RichText::new(format!("{name}:")).color(color));
}

/// Whether the fields are two to four numbers, as in `Vec3` or `UVec2`.
fn is_vector<'a>(fields: impl Iterator<Item = &'a dyn PartialReflect>) -> bool {
    let fields: Vec<&dyn PartialReflect> = fields.collect();
    let numbers = [
        TypeId::of::<f32>(),
        TypeId::of::<f64>(),
        TypeId::of::<i8>(),
        TypeId::of::<i16>(),
        TypeId::of::<i32>(),
        TypeId::of::<i64>(),
        TypeId::of::<isize>(),
        TypeId::of::<u8>(),
        TypeId::of::<u16>(),
        TypeId::of::<u32>(),
        TypeId::of::<u64>(),
        TypeId::of::<usize>(),
    ];
    (2..=4).contains(&fields.len())
        && fields.into_iter().all(|field| {
            field
                .get_represented_type_info()
                .is_some_and(|info| numbers.contains(&info.type_id()))
        })
}

/// A value of `variant` whose fields are their types' defaults, or `None` if
/// one of them doesn't reflect `Default`.
pub(super) fn default_variant(
    variant: &VariantInfo,
    registry: &TypeRegistry,
) -> Option<DynamicEnum> {
    let default = |type_id| {
        let default = registry.get_type_data::<ReflectDefault>(type_id)?;
        Some(default.default().into_partial_reflect())
    };
    let fields = match variant {
        VariantInfo::Unit(_) => DynamicVariant::Unit,
        VariantInfo::Tuple(variant) => {
            let mut fields = DynamicTuple::default();
            for field in variant.iter() {
                fields.insert_boxed(default(field.type_id())?);
            }
            DynamicVariant::Tuple(fields)
        }
        VariantInfo::Struct(variant) => {
            let mut fields = DynamicStruct::default();
            for field in variant.iter() {
                fields.insert_boxed(field.name(), default(field.type_id())?);
            }
            DynamicVariant::Struct(fields)
        }
    };
    Some(DynamicEnum::new(variant.name(), fields))
}
